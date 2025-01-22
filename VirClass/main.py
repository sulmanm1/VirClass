import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from Bio import SeqIO

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)
base_model = AutoModel.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True).to(device)

# Directory paths
Dir_path = "../../VirClassData"
fasta_path = f"{Dir_path}/seqs.fasta"
meta_path = f"{Dir_path}/meta.csv"

def read_fasta(filepath):
    return list(SeqIO.parse(filepath, "fasta"))

def map_records_to_classes(fasta_records, df):
    accession_to_class = df.set_index("Accession")["Class"].to_dict()
    data = []
    for record in fasta_records:
        accession = record.id
        if accession in accession_to_class:
            data.append((str(record.seq), accession_to_class[accession]))
    return data

class DNADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence, label = self.data[idx]
        inputs = self.tokenizer(
            sequence,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

class DNABERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super(DNABERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        # The model returns a tuple, so use the first element, which contains the hidden states.
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        # Some models ignore return_dict and return a tuple; if so, use outputs[0]
        if isinstance(outputs, dict):
            hidden_states = outputs["last_hidden_state"]
        else:
            hidden_states = outputs[0]
        # Pooling: take the mean across the sequence dimension
        pooled_output = torch.mean(hidden_states, dim=1)
        return self.classifier(pooled_output)

# Load metadata and FASTA
print("Loading metadata and FASTA...")
df = pd.read_csv(meta_path)
df['Class'] = df['Class'].astype('category').cat.codes
fasta_records = read_fasta(fasta_path)
data = map_records_to_classes(fasta_records, df)

# Split dataset
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

train_dataset = DNADataset(train_data, tokenizer)
val_dataset = DNADataset(val_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Initialize model, loss, and optimizer
print("Initializing model...")
num_classes = len(df['Class'].unique())
model = DNABERTClassifier("zhihan1996/DNABERT-S", num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=3e-5)
scaler = torch.cuda.amp.GradScaler()

def train_epoch(model, dataloader, criterion, optimizer, scaler):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def eval_epoch(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
    return total_loss / len(dataloader), correct / len(dataloader.dataset)

# Training loop
epochs = 3
print("Starting training...")
for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler)
    val_loss, val_acc = eval_epoch(model, val_loader, criterion)
    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

print("Training completed.")
