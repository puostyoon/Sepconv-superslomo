import os
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from Network import Network
from FrameDataset import FrameDataset

#Load the checkpoint file,
state_dict=torch.load("checkpoint/model_epoch1")

# Set the device to run on L GPU or CPU.
device=torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
# Get the "params" dictionary from the loaded state_dict
params = state_dict['params']

# Create the network.
net = Network().to(device)

# Load the trained generator weights.
net.load_state_dict(state_dict["net"])
print(net)

#instantiate Dataset and Create Dataloader
dataset = FrameDataset(is_train=False)
frame_dataloader = torch.utils.data.DataLoader(dataset,
                                               batch_size=3,
                                               shuffle=False)
# do the test
with torch.no_grad():
    for idx, frames in enumerate(frame_dataloader):
        output_frames = net.forward(frames[0].detach().to(device), frames[1].detach().to(device))
        frames_list0 = [frames[0][0].detach().cpu(), output_frames[0].detach().cpu(),
                               frames[2][0].detach().cpu()]
        frames_list1 = [frames[0][1].detach().cpu(), output_frames[1].detach().cpu(),
                               frames[2][1].detach().cpu()]
        frames_list2 = [frames[0][2].detach().cpu(), output_frames[2].detach().cpu(),
                               frames[2][2].detach().cpu()]
        sample_frames_grid = frames_list0 + frames_list1 + frames_list2
        plt.figure(figsize=(20, 20))
        plt.axis("off")
        plt.imshow(torch.permute(make_grid(sample_frames_grid, nrow=3, normalize=True), (1, 2, 0)))
        plt.savefig(os.path.join(params['root'],"test_results","Test result images %d"%(idx)))
        plt.close()
