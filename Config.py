# Hyper-parameters

"""
Root directory includes training dataset
Hierarchy of the root directory(when params["num_imgs_per_folder"] == 65, see below params dictionary):
|--root
     |--train
        |--video time line1
           |--0000.png
           |--0001.png
           |--. . .
           |--0064.png
        |--video time line2
            |--0000.png
            |--0001.png
            |--. . .
            |--0064.png
        |--. . .
     |--test
        |--video time line1
           |--0000.png
           |--0001.png
           |--. . .
           |--0064.png
        |--video time line2
            |--0000.png
            |--0001.png
            |--. .
            |--0064.png
        |--. . .


Each videotimeline folder should include 3 or more consecutive frames though network use 3 consecutive frames at once.
Or you can change FrameDataset.py so that you can use your own directory configuration.
"""

params = {
    "kernel_size": 51,
    "root": "C:/sepconv-puostyoon",
    "batch_size": 4,
    "num_epochs": 1,
    "save_epoch": 1,
    "num_imgs_per_folder" : 33
}
