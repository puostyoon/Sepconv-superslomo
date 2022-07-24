from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from Config import params
import random
import os

class FrameDataset(Dataset):
    """
    Read triplet set of frames at once in a directory.
    middle frame is the output and adjacent two frames are inputs
    """

    def __init__(self, is_train):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((768,768)), #this part can be removed
            # transforms.RandomCrop(128),
            # transforms.RandomHorizontalFlip(0.5),
            # transforms.RandomVerticalFlip(0.5),
            #identical random cropping and flipping should be applied to all of the 3 consecutive frames.
            #Thus we use custom transform function: random_crop_and_flip defined at below
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.root = params["root"] #root directory
        self.num_imgs_per_folder = params["num_imgs_per_folder"]
        self.is_train = is_train
        self.train_or_test = "train" if is_train == True else "test"
        self.video_folder_list = os.listdir(os.path.join(self.root, self.train_or_test))
    #end

    def random_crop_and_flip(self, imgs):
        """
        :param imgs: list of images, identical cropping and flipping will be applied to all the imgs in the list
        :return: transformed images

        random cropping and flipping works like the data augmentation
        """
        height, width = 128, 128
        top, left = random.randint(0, len(imgs[0][0])-128), random.randint(0, len(imgs[0][0][0])-128)
        flip_horizontal = True if random.random() > 0.5 else False
        flip_vertical = True if random.random() > 0.5 else False
        return_list = list()
        for idx, img in enumerate(imgs):
            temp = TF.crop(img, top, left, width, height)
            temp = TF.vflip(temp) if flip_vertical else temp
            temp = TF.hflip(temp) if flip_horizontal else temp
            return_list.append(temp)
        return return_list
    #end

    def __getitem__(self, index):
        #get index of the video folder
        folder_name = os.path.join(self.root, self.train_or_test,
                                   self.video_folder_list[index//(self.num_imgs_per_folder-2)])
        dir0 = os.path.join(folder_name, str("{:0>4d}".format(index%(self.num_imgs_per_folder-2))+".png"))
        dir1 = os.path.join(folder_name, str("{:0>4d}".format(index%(self.num_imgs_per_folder-2))+".png"))
        dir2 = os.path.join(folder_name, str("{:0>4d}".format(index%(self.num_imgs_per_folder-2))+".png"))
        frame0 = self.transform(Image.open(dir0))
        frame1 = self.transform(Image.open(dir1))
        frame2 = self.transform(Image.open(dir2))
        return self.random_crop_and_flip([frame0, frame1, frame2]) if self.is_train \
            else [TF.center_crop(frame0, output_size=[128, 128]),TF.center_crop(frame1, output_size=[128, 128]),
                  TF.center_crop(frame2, output_size=[128, 128])]
    #end

    def __len__(self):
        return len(self.video_folder_list)*(self.num_imgs_per_folder - 2)
    #end
#end