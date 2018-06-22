import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import os

class LinearRegressionModel(nn.Module):
    def __init__(self,insize,ousize):
        super(LinearRegressionModel, self).__init__()
        self.linear1 = nn.Linear(inp_dim,200)
        #self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(200,out_dim)
    def forward(self, x):
        out = F.relu(self.linear1(x))
        #out = self.dropout(out)
        out = self.linear2(out)
        return out

inp_dim=64
out_dim=4

test_model = LinearRegressionModel(inp_dim,out_dim)
test_model.load_state_dict(torch.load('singlebox_model_chk.pt'))

def main():
    #emojis = get_emojis()
    cap = cv2.VideoCapture(0)
    x, y, w, h = 300, 50, 350, 350

    while (cap.isOpened()):
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        a,b,c,d = Classify(test_model, img)
        print a,b,c,d
        #print(pred[0],pred[1],pred[2],pred[3])
        #img = overlay(img,emojis[pred[0]], 400, 250, 90, 90)
        x, y, w, h = 300, 50, 350, 350
        cv2.rectangle(img, (a,b), (a+c,b+d), (255,0,0), 2)
        cv2.imshow("Frame", img)
        k = cv2.waitKey(10)
        if k == 27:
            break

def Classify(model, image):
    processed = process_image(image)
    outputs=test_model(processed)
    print outputs
    o = outputs.data.numpy()
    print o
    o=o[0]
    return o[0],o[1],o[2],o[3]

def process_image(img):
    image_x = 8
    image_y = 8
    #img = cv2.imread(data_folder_path+sub_folders+'/'+file)
    print img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #print img.shape
    img = cv2.GaussianBlur(img, (7,7), 3)
    #print img.shape
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    #print img.shape
    ret, new = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #img = new
    #print new.shape
    #new = cv2.cvtColor(new,cv2.COLOR_GRAY2RGB)
    #print new.shape
    #cv2.imwrite('aa.png')
    img = cv2.resize(new, (image_x, image_y))
    img = np.array(img, dtype=np.float64)
    img = np.reshape(img, (image_x, image_y,-1))
    print img.shape
    img = (img.reshape(1, -1))
    print img.shape
    #ptLoader = transforms.Compose([transforms.ToTensor()])
    #img = ptLoader( img ).float()
    img = Variable( torch.FloatTensor(img), volatile=True  )
    print img.shape
    #print img
    return img

def overlay(image, emoji, x,y,w,h):
    emoji = cv2.resize(emoji, (w, h))
    try:
        image[y:y+h, x:x+w] = blend_transparent(image[y:y+h, x:x+w], emoji)
    except:
        pass
    return image

def blend_transparent(face_img, overlay_t_img):
    # Split out the transparency mask from the colour info
    overlay_img = overlay_t_img[:,:,:3] # Grab the BRG planes
    overlay_mask = overlay_t_img[:,:,3:]  # And the alpha plane

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))

Classify(test_model, np.zeros((8,8,3), dtype=np.uint8))
main()