import os
import time
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from Config import params
from FrameDataset import FrameDataset
from Network import Network
from Weight_initializer import init_weight
from LossFunction import LossFunction

#Use GPU if available.
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(device, "will be used")

#instantiate Network and initialize weights
net = Network().to(device)
net.apply(init_weight)

#define loss function and optimizer
criterion = LossFunction
optimizer = torch.optim.Adamax(net.parameters(), lr=0.001, betas=(0.9, 0.999))

#instantiate Dataset and Create Dataloader
dataset = FrameDataset(is_train=True)
frame_dataloader = torch.utils.data.DataLoader(dataset,
                                               batch_size=params["batch_size"],
                                               shuffle=True)

#save sample training images
sample_frames = next(iter(frame_dataloader))
"""
sample_frames = [torch.tensor(batch_size,3,128,128), torch.tensor(batch_size,3,128,128, 
                                                  torch.tensor(batch_size,3,128,128))"""
sample_frames_list0 = [sample_frames[0][0].detach(), sample_frames[1][0].detach(), sample_frames[2][0].detach()]
sample_frames_list1 = [sample_frames[0][1].detach(), sample_frames[1][1].detach(), sample_frames[2][1].detach()]
sample_frames_list2 = [sample_frames[0][2].detach(), sample_frames[1][2].detach(), sample_frames[2][2].detach()]
sample_frames_grid = sample_frames_list0+sample_frames_list1+sample_frames_list2
plt.figure(figsize=(20,20))
plt.axis("off")
plt.imshow(torch.permute(make_grid(sample_frames_grid, nrow = 3, normalize=True), (1,2,0)))
plt.savefig("Sample Training images")
plt.close()

#record time spent
start_time = time.time()

#List variables to store results of training.
img_list = []
losses = []

torch.save({
    "net": net.state_dict(),
    "optimizer": optimizer.state_dict(),
    "params": params
}, os.path.join(params["root"], "checkpoint/model_epoch%d" % (0)))

#start training
for epoch in range(params['num_epochs']):
    epoch_start_time = time.time()

    for i, frames in enumerate(frame_dataloader):

        #Transfer Data tensor to GPU/CPU (device)
        frame0 = frames[0].to(device)
        frame1 = frames[1].to(device)
        frame2 = frames[2].to(device)

        #run network
        s_frame = net.forward(frame0, frame2)

        #compute loss and gradients
        optimizer.zero_grad()
        loss = criterion(frame1, s_frame)
        loss.backward()
        optimizer.step()

        #Print progress of training.
        if i!=0 and i%50 == 0:
            print("[%d/%d][%d/%d]\tLoss: %.4f"%
                  (epoch+1, params["num_epochs"], i, len(frame_dataloader), loss.item()))

        #Save the losses for plotting
        losses.append(loss.item())

    epoch_time = time.time() - epoch_start_time
    print("Time taken for epoch %d: %.2fs"%(epoch+1, epoch_time))

    # Save network weights
    if (epoch + 1) % params["save_epoch"] == 0:
        torch.save({
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "params": params
        }, os.path.join(params["root"], "checkpoint/model_epoch%d" % (epoch + 1)))

    #Generate images to check the performance during training.
    if((epoch+1) == 1 or (epoch+1)%5 == 0):
        with torch.no_grad():
            interpolated_frames_per_epoch = net.forward(sample_frames[0][0:3].detach().to(device),
                                                        sample_frames[2][0:3].detach().to(device))
            frames_list_per_epoch0 = [sample_frames[0][0].detach().cpu(),
                                      interpolated_frames_per_epoch[0].detach().cpu(),
                                      sample_frames[2][0].detach().cpu()]
            frames_list_per_epoch1 = [sample_frames[0][1].detach().cpu(),
                                      interpolated_frames_per_epoch[1].detach().cpu(),
                                      sample_frames[2][1].detach().cpu()]
            frames_list_per_epoch2 = [sample_frames[0][2].detach().cpu(),
                                      interpolated_frames_per_epoch[2].detach().cpu(),
                                      sample_frames[2][2].detach().cpu()]
            frames_grid_per_epoch = frames_list_per_epoch0 + frames_list_per_epoch1 + frames_list_per_epoch2
        plt.figure(figsize=(20, 20))
        plt.axis("off")
        plt.imshow(torch.permute(make_grid(frames_grid_per_epoch, nrow=3, normalize=True), (1, 2, 0)))
        plt.savefig("result_at_epoch_%d" % (epoch + 1))
        plt.close()