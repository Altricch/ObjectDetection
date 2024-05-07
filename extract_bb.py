
import numpy as np
import cv2
import matplotlib.pyplot as plt


def extract_bb(labeltxt, imgpath, size = None):
    with open(labeltxt, 'r') as file:
        data = file.read()
    
    data = [float(x) for x in data.split()]

    # Extract label and coordinates
    label = data[0]
    coordinates = np.array(data[1:]).reshape(-1, 2)

    # Calculate bounding box
    x_min = np.min(coordinates[:, 0])
    y_min = np.min(coordinates[:, 1])
    x_max = np.max(coordinates[:, 0])
    y_max = np.max(coordinates[:, 1])
    
    img0 = cv2.imread(imgpath)
    height, width, _ = img0.shape
    
    if size is not None:
        height, width = size, size
        
    # Convert normalized coordinates to pixel coordinates
    x_min = int(x_min * width)
    y_min = int(y_min * height)
    x_max = int(x_max * width)
    y_max = int(y_max * height)
    
    return x_min, y_min, x_max, y_max


def show_img_bb(labelpath, imgpath, size=None):
    x_min, y_min, x_max, y_max = extract_bb(labelpath, imgpath, size)
    img0 = cv2.imread(imgpath)
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    if size is not None:
        img0 = cv2.resize(img0, dsize=(size,size), interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(img0, (x_min, y_min), (x_max, y_max), color=(255,0,0), thickness=2)
    plt.imshow(img0)
    plt.show() 


if __name__ == "__main__":
    img0 = cv2.imread("04Training_Data2/train/images/T1D0pa0dataset0.jpg")
    
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    x_min, y_min, x_max, y_max  = extract_bb("04Training_Data2/train/labels/T1D0pa0dataset0.txt", "04Training_Data2/train/images/T1D0pa0dataset0.jpg")
    print(x_min, y_min, x_max, y_max)
    cv2.rectangle(img0, (x_min, y_min), (x_max, y_max), color=(255,0,0), thickness=2)
    # cv2.imshow("Img", img0)   
    plt.imshow(img0)
    plt.show() 

