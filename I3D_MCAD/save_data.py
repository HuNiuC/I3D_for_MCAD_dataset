import numpy as np
import os
from PIL import Image
data_path ="/media/SeSaMe_NAS/data/Lixinhui/MCAD_jpeg_cut/"
to_path = "/media/SeSaMe_NAS/data/Lixinhui/I3D/data/MCAD_rgb/"

for ID_file in os.listdir(data_path):
    class_path = data_path + ID_file
    print ID_file
    for video_name in os.listdir(class_path):
        video_path = class_path +'/'+ video_name
        video_array = np.zeros((64,224,224,3))
        totalFrameNum = len(os.listdir(video_path))
        # more than 64 frames:
        if totalFrameNum > 63:
            start_frame = int((totalFrameNum - 64)/2)
            n = start_frame
            for index1 in range(64):
                frame = "frame%06d.jpg"%(n+1)
                frame_path = video_path + '/' + frame
                print frame_path
                if (os.path.isfile(frame_path)):
                    try :
                        frame  = Image.open(frame_path);
                    except IOError:
                        print("Cannot open the file ",frame_path)
                    np_img = np.array(frame)
                    np_img2 = (np_img -128.0)*1.0/128.0
                    video_array[index1,:,:,:]= np_img2
                    n = n+1;
        #less than 64 frames:
        else:
            start_frame = 0
            n = start_frame
            for index1 in range(64):
                if (index1 < totalFrameNum):
                    frame = "frame%06d.jpg"%(n+1)
                    frame_path = video_path + '/' + frame
                    if (os.path.isfile(frame_path)):
                        try :
                            frame  = Image.open(frame_path);
                        except IOError:
                            print("Cannot open the file ",frame_path)
                        np_img = np.array(frame)
                        np_img2 = (np_img -128.0)*1.0/128.0
                        video_array[index1,:,:,:]= np_img2
                        n = n+1;
                else :
                    frame = "frame%06d.jpg"%(n-totalFrameNum+1)
                    frame_path = video_path + '/' + frame
                    print frame_path
                    if (os.path.isfile(frame_path)):
                        try :
                            frame  = Image.open(frame_path);
                        except IOError:
                            print("Cannot open the file ",frame_path)
                        np_img = np.array(frame)
                        np_img2 = (np_img -128.0)*1.0/128.0
                        video_array[index1,:,:,:]= np_img2
                        n = n+1;
        save_path = to_path+ ID_file+ '/'+ video_name +'.npy' 
        print save_path 
        np.save(save_path,video_array)
     
