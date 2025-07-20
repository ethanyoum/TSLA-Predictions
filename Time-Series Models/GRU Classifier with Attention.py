class GRUWithAttentionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0,
                          bidirectional=bidirectional)

        self.attn = nn.Linear(hidden_size * self.num_directions, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size * self.num_directions)
        self.fc = nn.Linear(hidden_size * self.num_directions, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)

        attn_scores = self.attn(out).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(out * attn_weights.unsqueeze(-1), dim=1)

        out = self.dropout(context)
        out = self.norm(out)
        return self.fc(out)

def train_model(model, x_train, y_train, epochs=100, lr=0.0001, model_path=None, verbose=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Compute pos_weight for BCEWithLogitsLoss
    num_pos = y_train.sum().item()
    num_neg = y_train.shape[0] - num_pos
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x_train.to(device)).view(-1)
        loss = criterion(outputs, y_train.to(device).view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if verbose and (epoch + 1) % 5 == 0:
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            labels = y_train.cpu().numpy().astype(int).flatten()
            acc = accuracy_score(labels, preds)
            print(f"Epoch [{epoch+1}/{epochs}]  Loss: {loss.item():.6f}  Acc: {acc:.4f}")

    # Save model
    if model_path:
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    return model

# Evaluate Model
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, x_test, y_test, show_plot=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(x_test.to(device)).squeeze()
        probs = torch.sigmoid(logits)
        predicted_probs = probs.cpu().numpy()
        predicted_labels = (predicted_probs >= 0.5).astype(int)
        true_labels = y_test.cpu().numpy().astype(int).flatten()

    acc = accuracy_score(true_labels, predicted_labels)
    prec = precision_score(true_labels, predicted_labels)
    rec = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    cm = confusion_matrix(true_labels, predicted_labels)

    print(f"Accuracy:    {acc:.4f}")
    print(f"Precision:   {prec:.4f}")
    print(f"Recall:      {rec:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print("Confusion Matrix:\n", cm)

    if show_plot:
        plt.figure(figsize=(14, 6))
        plt.plot(predicted_probs, label='Predicted Probability', color='blue')
        for i, label in enumerate(true_labels):
            if label == 1:
                plt.axvline(i, color='orange', alpha=0.05)
        plt.title('TSLA Direction Prediction')
        plt.xlabel('Time Step')
        plt.ylabel('Probability / Direction')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.show()

    return predicted_probs, true_labels, acc, f1
