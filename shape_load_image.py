import numpy as np
import cv2
import glob
import os
from PIL import Image

def load_images_from_folder(folder):
    num_of_images = 0
    for filename in glob.glob(folder):
#        print(filename)
        idx= filename.find("synth_")
        part_type = filename[idx+6:idx+9]
        length = float(filename[idx+11:idx+14])
        width = float(filename[idx+16:idx+19])
        height= float(filename[idx+21:idx+24])
        partx = float(filename[idx+26:idx+31])
        party = float(filename[idx+33:idx+38])
        partz = float(filename[idx+40:idx+45])
        ang1  = float(filename[idx+47:idx+50])
        ang2  = float(filename[idx+52:idx+55])
        ang3  = float(filename[idx+57:idx+60])
        cam1= float(filename[idx+64:idx+68])
        cam2= float(filename[idx+72:idx+76])
        cam3= float(filename[idx+80:idx+84])
        cam4= float(filename[idx+88:idx+92])

        if part_type=="cyl":
            volume = 3.1415 * length * (width/2.0)**2
            y_one_hot = np.array([[1, 0, 0]])
        if part_type=="cub":
            volume = length * width * height
            y_one_hot = np.array([[0, 1, 0]])
        if part_type == "sph":
            volume = (4.0/3.0) * 3.1415 * (length / 2.0)**3
            y_one_hot = np.array([[0, 0, 1]])

        # print(part_type)
        # print(length, width, height, volume)
        # print(partx, party, partz)
        # print(ang1, ang2, ang3)
        # print(cam1, cam2, cam3, cam4)

#       print(filename)
        image = cv2.imread(filename)
#       cv2.imshow('Original image', image)
#       print(image[10,10])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#       print(gray[10,10])
#       cv2.imshow('Gray image', gray)

        if gray is not None:
            num_of_images += 1
            gray_r = gray.reshape(1, gray.shape[0], gray.shape[1], 1)

            if num_of_images == 1:
                sizex = gray.shape[0]
                sizey = gray.shape[1]
                x_image = gray_r.copy()
                x_numeric = np.array([[cam1, cam2, cam3, cam4]])
                y_shape = y_one_hot.copy()
                y_size = np.array([[length, width, height, ang1, ang2, ang3, volume]])
            else:
                if (gray.shape[0] == sizex) and (gray.shape[1] == sizey):
                    x_image = np.append(x_image, gray_r, axis=0)
                    x_numeric = np.append(x_numeric, np.array([[cam1, cam2, cam3, cam4]]), axis=0)
                    y_shape = np.append(y_shape, y_one_hot, axis=0)
                    y_size = np.append(y_size,
                                       np.array([[length, width, height, ang1, ang2, ang3, volume]]), axis=0)
                else:
                    print(sizex, gray.shape[0])
                    print(sizey, gray.shape[1])
                    print('Image number', num_of_images, ' : Size mis-match')
                    exit()
        #print("Shape of x :", x.shape, end='\n         ')
        #print("Shape of y :", y.shape, end='\n         ')
        #print(str(num_of_images)+' '+filename+' '+str(gray.shape))

    print("Data read from ", str(folder), end='\n      ')
    print("Total number of images read =", x_image.shape[0], end='\n      ')
    print("Pixels in each image =", sizex, ' * ', sizey, end='\n      ')
    print("Total number of pixels", sizex * sizey, end='\n         ')

    return x_image, x_numeric, y_shape, y_size, sizex, sizey

def create_data_sets():
    #read synthetic data set
    dirname_regression_data = \
        "/home/kprasad/synthetic_shape_data_all/"
    dirname = os.listdir(dirname_regression_data)

    print("Number of directories :", len(dirname),"\n")

    num_of_dir = 0
    for dir in dirname:
        num_of_dir += 1
        folder_name = str(dirname_regression_data)+str(dir)+str('/*.png')

        #print(folder_name)

        (x_image, x_numeric, y_shape, y_size, sizex, sizey) = \
         load_images_from_folder(folder_name)

        x_set = x_image.copy()
        y_set = y_shape.copy()

        if num_of_dir == 1:
            print("Creating first dataset\n")
            x = x_set.copy()
            y = y_set.copy()
        else:
            print("Appending datasets\n")
            x = np.append(x, x_set, axis=0)
            y = np.append(y, y_set, axis=0)

    #rand_pic=np.random.randint(0,m)
    #arr=x[rand_pic,:].reshape(sizex, sizey)
    #im = Image.fromarray(arr)
#    im.show()

    #arr = x[rand_pic, 0:num_pix//2].reshape(sizex//2, sizey)
    #im = Image.fromarray(arr)
#    im.show()

    #arr1 = arr[:,0:sizey//2]
    #im = Image.fromarray(arr1)
#    im.show()

    return x, y, sizex, sizey
