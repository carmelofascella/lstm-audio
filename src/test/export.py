## This script will export a trained model 
## in torchscript format 

import torch
import sys
sys.path.insert(1, 'src/')
import lstm_models

# replace saved_pth_path with your own.
saved_pth_path = 'runs/Jan22_12-13-57_Carmelos-Macbook.local ssl-32-unit LSTM model with 32 hidden units/saved_models/lstm_size_32_epoch_0_loss_0.1876.pth'

export_pt_path = 'dist_32.pt'

# model = myk_models.SimpleLSTM(hidden_size=32)
model = torch.load(saved_pth_path)

# this is possibly not necessary but in case
# the model does anything like dropout layers 
# that only operate when training 
model.eval()

scripted_model = torch.jit.script(model)

torch.jit.save(scripted_model, export_pt_path)

# now test it 
in_test = torch.rand((1, 10, 1)) 
model2 = torch.jit.load(export_pt_path)

out1 = model.forward(in_test) 
out2 = model2.forward(in_test) 

assert(torch.equal(out1, out2)), "Somethings wrong: output is different"
