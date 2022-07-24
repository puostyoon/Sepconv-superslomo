"""
set root variable in line 44 to be your root directory.
please don't change the file name of this code, or you can change the line 46 to be the file name
this code change hierarchy such as
--root
          -- type of video1
             -- video time line1
                --0000.png
                --0001.png
                      .
                      .
                      .
             -- video time line2
                --0000.png
                --0001.png
                      .
                      .
                      .
          -- type of video2
                  .
                  .


to be:

--root
        --video time line1
           --0000.png
           --0001.png
                 .
                 .
                 .
        --video time line2
            --0000.png
            --0001.png
                 .
                 .
       
"""


import os

root = "C:\\sepconv-puostyoon\\test"
list0 = os.listdir(root)
list0.remove("folder_hierarchy_change.py")
folder_name_list = [os.listdir(os.path.join(root, i)) for i in list0]
for i0, i in enumerate(folder_name_list):
    for j0, j in enumerate(i):
        idx = i0*len(i)+j0
        print(len(folder_name_list))
        print(idx)
        os.rename(os.path.join(root,list0[folder_name_list.index(i)],j), 'f'+ str(idx) )
