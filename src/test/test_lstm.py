import torch
import numpy as np
from scipy.io.wavfile import write

#ßet random seed so it generates the same network parameters
torch.manual_seed(21)

# synthesize a 400Hz sine at 44.100Hz sample rate
freq = 400
sine = np.sin(np.arange(0, 2*np.pi*freq, (np.pi * freq * 2) / 44100), dtype=np.float32)

#reshape so each sample has its own box
sine = np.reshape(sine, (len(sine), 1))

#simple LSTM
input_size = 1
hidden_size = 1
num_layers = 1
lstm = torch.nn.LSTM(input_size, hidden_size, num_layers)

input = torch.rand(1, 1)
h0 = torch.rand(1, 1)
c0 = torch.rand(1, 1)

traced_lstm = torch.jit.trace(lstm, (input, (h0, c0)))
traced_lstm.save('my_lstm.pt')

#test input 
in_t = torch.tensor(sine)

output, state = lstm.forward(input)
# print(output)

output, state = traced_lstm.forward(input, state)
print(output)


write('sine_400.wav', 44100, sine * 0.5)
write('sine_400-lstm.wav', 44100, output.detach().numpy())