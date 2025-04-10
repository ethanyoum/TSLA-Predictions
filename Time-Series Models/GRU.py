# Construction of GRU
class GRU(nn.Module):
  def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
    super(GRU, self).__init__()
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first = True)
    self.fc = nn.Linear(hidden_dim, output_dim)

  def forward(self, x):
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
    out, (hn) = self.gru(x, (h0.detach()))
    out = self.fc(out[:, -1, :])
    return out

gru_test = GRU(input_dim = input_dim, hidden_dim = hidden_dim, num_layers = num_layers, output_dim = output_dim)

# Specify Loss function & Optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(gru.parameters(), lr = 0.01)

## Train GRU
def train_model_gru(model, x_train, y_train, num_epochs):
  train_losses = np.zeros(num_epochs)
  start_time = time.time()

  for epoch in range(num_epochs):
    y_train_pred = model(x_train)
    loss = criterion(y_train_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Epoch ", epoch + 1, "MSE: ", loss.item())
    train_losses[epoch] = loss.item()

  training_time = time.time() - start_time
  print("Training time: {}".format(training_time))

  return y_train_pred, train_losses, training_time

gru_train_pred, gru_training_losses, gru_training_time = train_model_gru(gru, x_train, y_train, num_epochs = 100)

# Evaluation
def evaluation_gru(model, x_test, y_test, y_train, y_train_pred):
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

gru_test_pred, gru_results = evaluation_gru(gru, x_test, y_test, y_train, gru_train_pred)
