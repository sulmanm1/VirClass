import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from Bio import SeqIO
import random
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import torch.nn.functional as F

# --------------------- #
#    Argument Parser    #
# --------------------- #
argparser = argparse.ArgumentParser()
argparser.add_argument("--data_dir", type=str, default=f"{Path.home()}/.cache/ViruLink/class_db")
argparser.add_argument("--model_path", type=str, default="best_model.pth", help="Path to the saved model")
argparser.add_argument("--skip_training", action="store_true", help="Skip training and use the saved model")
argparser.add_argument("--samples_per_class", type=int, default=5000, help="Number of samples per class for training")
args = argparser.parse_args()

# --------------------- #
#   Check for GPU/CPU   #
# --------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------- #
#    Load DNABERT-S     #
# --------------------- #
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)
base_model = AutoModel.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True).to(device)

# --------------------- #
#   Directory & Paths   #
# --------------------- #
Dir_path = args.data_dir
fasta_path = f"{Dir_path}/filtered_sequences.fasta"
meta_path = f"{Dir_path}/merged_df.csv"

# --------------------- #
#   FASTA & Metadata    #
# --------------------- #
def read_fasta(filepath):
    return list(SeqIO.parse(filepath, "fasta"))

def calculate_genome_size_per_class(fasta_records, df):
    """
    Calculate total genome size (sum of lengths) for each class
    based on the assemblies in the given DataFrame.
    """
    class_genome_sizes = {}
    for class_label in df['Class'].unique():
        assemblies = df[df['Class'] == class_label]['Assembly']
        class_records = [r for r in fasta_records if r.id in assemblies.values]
        total_size = sum(len(r.seq) for r in class_records)
        class_genome_sizes[class_label] = total_size
    return class_genome_sizes

# --------------------- #
#  Random Fragmenting   #
# --------------------- #
def sample_sequences(sequence, sample_size, fragment_sizes):
    """
    Randomly sample fragments (and their reverse complements) from a long sequence.
    """
    sampled_sequences = []
    for size in fragment_sizes:
        if len(sequence) < size:
            continue
        # Divide sample_size among different fragment sizes
        # so each size gets (sample_size // len(fragment_sizes)) fragments.
        for _ in range(sample_size // len(fragment_sizes)):
            start = random.randint(0, len(sequence) - size)
            fragment = sequence[start:start + size]
            sampled_sequences.append(fragment)
            # Reverse complement
            reverse_complement = fragment.translate(str.maketrans("ACGT", "TGCA"))[::-1]
            sampled_sequences.append(reverse_complement)
    return sampled_sequences

def map_and_sample_data(fasta_records, df, samples_per_class):
    """
    For each class, combine all its sequences into one large string,
    then randomly sample fragments from it. 
    """
    class_to_assemblies = df.groupby('Class')['Assembly'].apply(list).to_dict()
    class_genome_sizes = calculate_genome_size_per_class(fasta_records, df)

    sampled_data = []
    fragment_sizes = [500, 1000, 1500, 2000]

    for class_label, assemblies in class_to_assemblies.items():
        class_records = [r for r in fasta_records if r.id in assemblies]
        combined_sequence = "".join(str(r.seq) for r in class_records)

        if len(combined_sequence) == 0:
            continue

        total_genome_size = class_genome_sizes[class_label]

        # Adjust sample size if total genome size is small
        adjusted_sample_size = samples_per_class
        while adjusted_sample_size * 2 > total_genome_size and adjusted_sample_size > 0:
            adjusted_sample_size //= 2

        if adjusted_sample_size == 0:
            # If this class can't yield fragments given its genome size, skip
            continue

        sampled_sequences = sample_sequences(combined_sequence, adjusted_sample_size, fragment_sizes)
        sampled_data.extend([(seq, class_label) for seq in sampled_sequences])

    return sampled_data

# --------------------- #
#     Torch Dataset     #
# --------------------- #
class DNADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        """
        data: list of (sequence_string, label)
        """
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

# --------------------- #
#   Attention Pooling   #
# --------------------- #
class AttentionPool(nn.Module):
    """
    A simple trainable attention pooling layer.
    Learns a context vector to score each token embedding.
    """
    def __init__(self, hidden_dim):
        super(AttentionPool, self).__init__()
        # A trainable parameter for scoring
        self.context_vector = nn.Parameter(torch.randn(hidden_dim))
        # Optional projection:
        # self.projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states, mask=None):
        """
        hidden_states: (batch_size, seq_len, hidden_dim)
        mask: (batch_size, seq_len), 1 for valid tokens, 0 for padding
        """
        # If you wanted a projection + activation:
        # projected = torch.tanh(self.projection(hidden_states))
        # scores = torch.einsum('btd,d->bt', projected, self.context_vector)
        
        scores = torch.einsum('btd,d->bt', hidden_states, self.context_vector)  # (B, T)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)  # (B, T)
        pooled_output = torch.einsum('bt,btd->bd', attn_weights, hidden_states)
        return pooled_output

# --------------------- #
#  DNABERT Classifier   #
# --------------------- #
class DNABERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super(DNABERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        # Replace default pooling with an attention-based pooling
        self.attention_pool = AttentionPool(hidden_dim=768)

        # A deeper classifier head
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0]  # shape (B, T, 768)

        # Apply custom attention pooling
        pooled_output = self.attention_pool(hidden_states, mask=attention_mask)
        # Classifier
        logits = self.classifier(pooled_output)
        return logits

# --------------------- #
#   Load Data & Split   #
# --------------------- #
print("Loading metadata and FASTA...")
df = pd.read_csv(meta_path)

# Convert class labels to integer codes
df['Class'] = df['Class'].astype('category')
class_labels = df['Class'].cat.categories  # Original class names
df['Class'] = df['Class'].cat.codes       # Convert to integer codes

fasta_records = read_fasta(fasta_path)

# Split assemblies into train/val
assemblies = df['Assembly'].unique()
train_assemblies, val_assemblies = train_test_split(assemblies, test_size=0.1, random_state=42)

train_df = df[df['Assembly'].isin(train_assemblies)]
val_df = df[df['Assembly'].isin(val_assemblies)]

# --------------------- #
#   Scale the #samples  #
# --------------------- #
train_genome_sizes = calculate_genome_size_per_class(fasta_records, train_df)
val_genome_sizes   = calculate_genome_size_per_class(fasta_records, val_df)

train_total_genome_size = sum(train_genome_sizes.values())
val_total_genome_size   = sum(val_genome_sizes.values())

if train_total_genome_size == 0:
    train_total_genome_size = 1  # Avoid division by zero

val_ratio = val_total_genome_size / train_total_genome_size
train_samples_per_class = args.samples_per_class
val_samples_per_class   = int(args.samples_per_class * val_ratio)

print(f"Training total genome size: {train_total_genome_size}")
print(f"Validation total genome size: {val_total_genome_size}")
print(f"Ratio for validation set: {val_ratio:.3f}")
print(f"Train samples/class: {train_samples_per_class}")
print(f"Val   samples/class: {val_samples_per_class}")

# --------------------- #
#   Map & Sample Data   #
# --------------------- #
train_data = map_and_sample_data(fasta_records, train_df, train_samples_per_class)
val_data   = map_and_sample_data(fasta_records, val_df, val_samples_per_class)

print(f"Number of training samples: {len(train_data)}")
print(f"Number of validation samples: {len(val_data)}")

# Create datasets and dataloaders
train_dataset = DNADataset(train_data, tokenizer)
val_dataset   = DNADataset(val_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=40, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=40, shuffle=False)

# --------------------- #
#     Model & Optims    #
# --------------------- #
num_classes = len(df['Class'].unique())
model = DNABERTClassifier("zhihan1996/DNABERT-S", num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=3e-5)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
scaler = torch.cuda.amp.GradScaler()

# --------------------- #
#    Train & Evaluate   #
# --------------------- #
def train_epoch(model, dataloader, criterion, optimizer, scaler):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def eval_epoch(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            with torch.cuda.amp.autocast():
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / len(dataloader.dataset)
    return avg_loss, accuracy, all_preds, all_labels

# If we already have a saved model and skip_training was requested
if os.path.exists(args.model_path) and args.skip_training:
    print(f"Loading model from {args.model_path}...")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print("Evaluating model to generate confusion matrix...")
    _, val_acc, all_preds, all_labels = eval_epoch(model, val_loader, criterion)
    print(f"Validation Accuracy: {val_acc:.4f}")
else:
    # --------------------- #
    #   Training Loop      #
    # --------------------- #
    epochs = 10
    best_val_acc = 0.0

    print("Starting training...")
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc, all_preds, all_labels = eval_epoch(model, val_loader, criterion)

        # Step the scheduler
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.model_path)
            print(f"New best model saved with accuracy: {val_acc:.4f}")

        # Confusion Matrix
        print("Generating Confusion Matrix...")
        conf_matrix = confusion_matrix(all_labels, all_preds, normalize="true")
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_labels)

        # Make the figure bigger so labels fit
        fig, ax = plt.subplots(figsize=(20, 16))
        disp.plot(cmap="Blues", ax=ax, values_format=".2f")
        plt.xticks(rotation=90)
        plt.title(f"Confusion Matrix (Epoch {epoch + 1})")
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.tight_layout()
        plt.savefig(f"confusion_matrix_epoch_{epoch + 1}.png")
        plt.close(fig)

print("Done.")
