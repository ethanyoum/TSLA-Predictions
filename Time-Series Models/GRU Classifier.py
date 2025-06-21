class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.dropout(out[:, -1, :])
        out = self.norm(out)
        return self.sigmoid(self.fc(out))

# Train model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

def train_model(model, x_train, y_train, epochs=30, lr=0.00009, model_path=None, verbose=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x_train.to(device)).squeeze()
        targets = y_train.to(device).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if verbose and (epoch + 1) % 5 == 0:
            preds_binary = (outputs.detach().cpu().numpy() >= 0.5).astype(int)
            true_binary = y_train.cpu().numpy().astype(int).flatten()
            acc = accuracy_score(true_binary, preds_binary)
            print(f"Epoch [{epoch+1}/{epochs}]  Loss: {loss.item():.6f}  Acc: {acc:.4f}")

    # Save model
    if model_path:
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    return model

# Evaluate model
def evaluate_model(model, x_test, y_test, show_plot=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(x_test.to(device)).squeeze()
        predicted_probs = outputs.cpu().numpy()
        predicted_labels = (predicted_probs >= 0.5).astype(int)
        true_labels = y_test.cpu().numpy().astype(int).flatten()

    # Metrics
    acc = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    cm = confusion_matrix(true_labels, predicted_labels)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Confusion Matrix:\n", cm)

    if show_plot:
      plt.figure(figsize=(14, 6))
    
      plt.plot(predicted_probs, label='Predicted Probability', color='blue', linewidth=2)
    
      for i, val in enumerate(actual):
          if val == 1:
              plt.axvspan(i - 0.5, i + 0.5, color='orange', alpha=0.3)

      plt.title("TSLA Direction Prediction")
      plt.xlabel("Time Step")
      plt.ylabel("Probability / Direction")
      plt.ylim([-0.1, 1.1])
      plt.legend(loc="upper left")
      plt.grid(True)
      plt.tight_layout()
      plt.show()

    return predicted_probs, true_labels, acc, f1
