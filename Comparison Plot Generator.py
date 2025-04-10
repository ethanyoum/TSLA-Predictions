# Inverse transform the values to plot the graph comparing actual vs predicted
original = pd.DataFrame(scaler.inverse_transform(y_train.detach().numpy()))
predict_rnn = pd.DataFrame(scaler.inverse_transform(rnn_train_pred.detach().numpy()))
predict_lstm = pd.DataFrame(scaler.inverse_transform(lstm_train_pred.detach().numpy()))
predict_gru = pd.DataFrame(scaler.inverse_transform(gru_train_pred.detach().numpy()))
predict_cm = pd.DataFrame(scaler.inverse_transform(cm_train_pred.detach().numpy()))

# Plotting the comparison actual vs predicted
def plot_training_results(predict, original, train_losses, model_name):
    sns.set_style("darkgrid")
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.subplot(1, 2, 1)
    ax = sns.lineplot(x = original.index, y = original[0], label="Data", color='royalblue')
    ax = sns.lineplot(x = predict.index, y = predict[0], label="Training Prediction (" + model_name +")" , color='tomato')
    ax.set_title(ticker+' Stock price', size = 14, fontweight='bold')
    ax.set_xlabel("Days", size = 14)
    ax.set_ylabel("Price (USD)", size = 14)
    ax.set_xticklabels('', size=10)

    plt.subplot(1, 2, 2)
    ax = sns.lineplot(data=train_losses, color='royalblue')
    ax.set_xlabel("Epoch", size = 14)
    ax.set_ylabel("Loss", size = 14)
    ax.set_title("Training Loss", size = 14, fontweight='bold')
    fig.set_figheight(6)
    fig.set_figwidth(16)

RNN_com_plot = plot_training_results(predict_rnn, original, rnn_train_losses, 'RNN')
LSTM_com_plot = plot_training_results(predict_lstm, original, rnn_train_losses, 'LSTM')
GRU_com_plot = plot_training_results(predict_gru, original, rnn_train_losses, 'GRU')
