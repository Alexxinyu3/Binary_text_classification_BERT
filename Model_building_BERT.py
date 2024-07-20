import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR = OUTPUT_DIR / "plot"
WEIGHTS_DIR = OUTPUT_DIR / "weights"

# 1.load the data
file_path = 'D:/Desktop/study in France/ESIGELEC-study/Intership/IPSOS/cleaned_data_for_model.xlsx'
df = pd.read_excel(file_path).dropna()  # Drop NaN
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

# 2.Split the data into Training and evaluating Sets
train_df, eval_df = train_test_split(df, test_size=0.3, random_state=42)

# 3.Tokenize and Encode the Text Data Using BERT Tokenizer
# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# Tokenize and encode the data
def encode_data(texts, labels, max_length=128):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0), torch.tensor(labels)


train_inputs, train_masks, train_labels = encode_data(train_df.cleaned_text.values, train_df.sentiment.values)
eval_inputs, eval_masks, eval_labels = encode_data(eval_df.cleaned_text.values, eval_df.sentiment.values)

### 4.Create a PyTorch Dataset and DataLoader
# Create the DataLoader for our training set
train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
train_dataloader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),
    batch_size=32
)

# Create the DataLoader for our eval set
eval_dataset = TensorDataset(eval_inputs, eval_masks, eval_labels)
eval_dataloader = DataLoader(
    eval_dataset,
    sampler=SequentialSampler(eval_dataset),
    batch_size=32
)

### 5.Fine-tune a Pre-trained BERT Model for Text Classification
# Load the BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=1,
    output_attentions=False,
    output_hidden_states=False
)

# Move model to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Set up the optimizer
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

### 6. Train the Model

# Number of training epochs
epochs = 1

# Total number of training steps
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

best_acc, train_losses, eval_losses, accuracies = 0, [], [], []
# Training loop
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    for step, batch in enumerate(train_dataloader):
        b_input_ids, b_input_mask, b_labels = batch
        b_input_ids = b_input_ids.to(device)
        b_input_mask = b_input_mask.to(device).to(torch.float32)
        b_labels = b_labels.to(device).to(torch.float32)

        model.zero_grad()

        # Forward pass
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss.to(torch.float32)
        total_train_loss += loss.item()

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and learning rate
        optimizer.step()
        scheduler.step()
        print(f"> Step {step}, Loss: {loss.item()}")

    # Epoch finished, calculate train loss
    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f'>> Epoch {epoch + 1}, Train Loss: {avg_train_loss}')
    train_losses.append(avg_train_loss)

    # Evaluation
    model.eval()
    total_eval_loss = 0
    predictions, true_labels = [], []

    for batch in eval_dataloader:
        b_input_ids, b_input_mask, b_labels = batch
        b_input_ids = b_input_ids.to(device)
        b_input_mask = b_input_mask.to(device)

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

        total_eval_loss += outputs.loss.item()
        logits = outputs.logits
        # predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
        predictions.extend((logits.squeeze() >= 0.5).long().cpu().numpy().tolist())
        true_labels.extend(b_labels.cpu().numpy())

    avg_eval_loss = total_eval_loss / len(eval_dataloader)
    eval_losses.append(avg_eval_loss)
    accuracy = accuracy_score(true_labels, predictions)
    accuracies.append(accuracy)
    print(
        f'>> EVAL == Accuracy: {accuracy}  '
        f'Eval Loss: {avg_eval_loss}'
    )
    torch.save(model.state_dict(), str(WEIGHTS_DIR / f'best_e{epoch + 1}.pth')) if accuracy > best_acc else None

# TODO: Use matplotlib.pyplot draw -- Train loss curve, eval loss curve, accuracy curve
print(
    f"Train Loss: {train_losses}\n"
    f"Eval Loss: {eval_losses}\n"
    f"Accuracies: {accuracies}\n"
    f"Best Accuracy: {best_acc}"
)

### 7. Evaluate the Model on the eval Set

model.eval()
predictions, true_labels = [], []

for batch in eval_dataloader:
    b_input_ids, b_input_mask, b_labels = batch
    b_input_ids = b_input_ids.to(device)
    b_input_mask = b_input_mask.to(device)

    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

    logits = outputs.logits
    # predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
    predictions.extend((logits.squeeze() >= 0.5).long().cpu().numpy().tolist())
    true_labels.extend(b_labels.cpu().numpy())

accuracy = accuracy_score(true_labels, predictions)
print(f'Accuracy: {accuracy}')
print(classification_report(true_labels, predictions))
torch.save(model.state_dict(), str(WEIGHTS_DIR / 'final.pth'))

# 8. Plot Losses & Accuracies
plt.figure()
plt.plot(train_losses, marker='o', linestyle='-', color='b')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig(str(PLOT_DIR / "train_loss.png"))

plt.figure()
plt.plot(eval_losses, marker='o', linestyle='-', color='b')
plt.title('Evaluating Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig(str(PLOT_DIR / "eval_loss.png"))

plt.figure()
plt.plot(accuracies, marker='x', linestyle='--', color='r')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig(str(PLOT_DIR / "acc.png"))
