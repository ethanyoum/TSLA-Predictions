# Construction of RNN model
class RNN(nn.Module):
  def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
    super(RNN, self).__init__()
    self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first = True)
    self.fc = nn.Linear(hidden_dim, output_dim)

  def forward(self, x, hidden):
    r_output, hidden = self.rnn(x, hidden)
    output = self.fc(r_output[:, -1, :])
    return output, hidden

# Test and check dimensions
test_rnn = RNN(input_dim=1, output_dim=1, hidden_dim=10, num_layers=2)

# Generate evenly spaced, test data pts
test_input= x_train[0:64,:,:]
test_input = test_data.unsqueeze(0) # give it a batch_size of 1 as first dimension
print('Input size: ', test_input.size())

# Test out rnn sizes
test_out, test_h = test_rnn(test_input, None)
print('Output size: ', test_out.size())
print('Hidden state size: ', test_h.size())

## Train RNN
# Define parameters
input_dim = 1
output_dim = 1
hidden_dim = 32
num_layers = 2

rnn = RNN(input_dim, output_dim, hidden_dim, num_layers)
print(rnn)

# Specify Loss function & Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)

# Training
def train_model(model, x_train, y_train, num_epochs):
  start_time = time.time()
  train_losses = np.zeros(num_epochs)

  for epoch in range(num_epochs):
    total_train_losses = 0.0
    hidden = None
    y_train_pred, hidden = model(x_train, hidden)
    loss = criterion(y_train_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch', epoch + 1, 'MSE:', loss.item())
    train_losses[epoch] = loss.item()
  training_time = time.time() - start_time
  print('training time: {}'.format(training_time))

  return y_train_pred, train_losses, training_time

rnn_train_pred, rnn_train_losses, rnn_training_time = train_model(rnn, x_train, y_train, num_epochs = 100)

# Evaluation
import math, time
from sklearn.metrics import mean_squared_error

def evaluate_model(model,x_test,y_test,y_train,y_train_pred):
  result=[]

  hidden = None
  y_test_pred, hidden = model(x_test, hidden)

  # Invert predictions
  y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
  y_test_orig = scaler.inverse_transform(y_test.detach().numpy())
  y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
  y_train_orig = scaler.inverse_transform(y_train.detach().numpy())

  # Calculate root mean squared error
  trainScore = math.sqrt(mean_squared_error(y_train_orig[:,0], y_train_pred[:,0]))
  print('Train Score: %.2f RMSE' % (trainScore))
  testScore = math.sqrt(mean_squared_error(y_test_orig[:,0], y_test_pred[:,0]))
  print('Test Score: %.2f RMSE' % (testScore))
  result.append(trainScore)
  result.append(testScore)
  result.append(rnn_training_time)
  return y_test_pred, result

rnn_test_pred, rnn_result=evaluate_model(rnn,x_test,y_test,y_train,rnn_train_pred)
