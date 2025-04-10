# COnstruction of Customized GRU model
class Customized_model(nn.Module):
  def __init__(self, input_dim, cnn_filters, hidden_dim, num_layers, output_dim):
    super(Customized_model, self).__init__()
    # CNN for feature extraction
    self.conv1 = nn.Conv1d(in_channels = input_dim, out_channels = cnn_filters, kernel_size = 3, padding = 1)
    self.relu = nn.ReLU()
    self.gru = nn.GRU(cnn_filters, hidden_dim, num_layers, batch_first = True)

    # Integrate Bahdanau attention mechanism
    self.attention = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
    self.v = nn.Linear(hidden_dim, 1, bias = False)
    self.fc = nn.Linear(hidden_dim, output_dim)

  def forward(self, x):
    x = self.conv1(x.permute(0, 2, 1))
    x = self.relu(x)
    x = x.permute(0, 2, 1)

    gru_out, hidden = self.gru(x)

    hidden = hidden[-1].unsqueeze(1).repeat(1, gru_out.size(1), 1)
    concat = torch.cat((gru_out, hidden), dim = 2)
    attention_scores = torch.tanh(self.attention(concat))
    attention_weights = torch.softmax(self.v(attention_scores).squeeze(-1), dim = 1)
    context_vector = gru_out[:, -1, :]
    context_vector = torch.sum(gru_out * attention_weights.unsqueeze(-1), dim = 1)

    out = self.fc(context_vector)
    return out

# New hyperparameter
cnn_filters = 16

test_cm = Customized_model(input_dim = input_dim, cnn_filters = cnn_filters, hidden_dim = hidden_dim, num_layers = num_layers, output_dim = output_dim)
print(test_cm)

# Specify Loss function & Optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(cm.parameters(), lr = 0.01)

# Train the model
def train_customized_model(model, x_train, y_train, num_epochs, scheduler = None, patience = 5):
  train_losses = np.zeros(num_epochs)
  best_loss = float('inf')
  epochs_no_improvement = 0
  start_time = time.time()

  for epoch in range(num_epochs):
    y_train_pred = model(x_train)
    loss = criterion(y_train_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if scheduler:
      scheduler.step(loss)

    train_losses[epoch] = loss.item()

    if loss.item() < best_loss:
      best_loss = loss.item()
      epochs_no_improvement = 0
    else:
      epochs_no_improvement += 1

    print("Epoch ", epoch + 1, "MSE: ", loss.item())

    if epochs_no_improvement >= patience:
      print(f"Early stop triggered after {epoch + 1} epochs")
      break

  training_time = time.time() - start_time
  print("Training time: {}".format(training_time))

  return y_train_pred, train_losses, training_time

# Set scheduler for early stopping
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.5, patience = 5)

cm_train_pred, cm_training_losses, cm_training_time = train_customized_model(cm, x_train, y_train, num_epochs = 100, scheduler = scheduler, patience = 5)

# Validation
def evaluation_customized_model(model, x_test, y_test, y_train, y_train_pred):
  results = []
  y_test_pred = model(x_test)

  # Invert values
  y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
  y_train_orig = scaler.inverse_transform(y_train.detach().numpy())
  y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
  y_test_orig = scaler.inverse_transform(y_test.detach().numpy())

  # Calculate Root Mean Squared Error
  trainScore = math.sqrt(mean_squared_error(y_train_orig[:, 0], y_train_pred[:, 0]))
  print('Train Score: %.2f RMSE' % (trainScore))
  testScore = math.sqrt(mean_squared_error(y_test_orig[:, 0], y_test_pred[:, 0]))
  print('Test Score: %.2f RMSE' % (testScore))
  results.append(trainScore)
  results.append(testScore)
  results.append(gru_training_time)
  return y_test_pred, results
