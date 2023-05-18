import numpy as np
import math
import torch
from PIL import Image

def psnr(target, ref):
	#将图像格式转为float64
    target_data = np.array(target, dtype=np.float64)
    ref_data = np.array(ref,dtype=np.float64)
    # 直接相减，求差值
    diff = ref_data - target_data
    # 按第三个通道顺序把三维矩阵拉平
    diff = diff.flatten('C')
    # 计算MSE值
    rmse = math.sqrt(np.mean(diff ** 2.))
    # 精度
    eps = np.finfo(np.float64).eps
    if(rmse == 0):
        rmse = eps 
    return 20*math.log10(255.0/rmse)

def fill_image(x, block_size):
    h = x.size()[1]
    h_lack = 0
    w = x.size()[2]
    w_lack = 0
    if h % block_size != 0:
        h_lack = block_size - h % block_size
        temp_h = torch.zeros(3, h_lack, w)
        h = h + h_lack
        x = torch.cat((x, temp_h), 1)

    if w % block_size != 0:
        w_lack = block_size - w % block_size
        temp_w = torch.zeros(3, h, w_lack)
        w = w + w_lack
        x = torch.cat((x, temp_w), 2)
    return x, h ,w

def read_img(path):
    image = Image.open(path).convert("RGB")
    image = np.array(image)
    image = (image/127.5 - 1.0).astype(np.float32)
    image = np.transpose(image,(2,0,1))
    image = torch.from_numpy(image)
    return image

def make_batch(image, block_size, device, channels = 3):
    x ,h ,w = fill_image(image, block_size)
    # print('********',x.size())

    # x = torch.unsqueeze(x, 0)
    x = torch.unsqueeze(x, 0)

    idx_h = range(0, h, block_size)
    idx_w = range(0, w, block_size)
    num_batches = h * w // (block_size ** 2)

    batchs = torch.zeros(num_batches, channels, block_size, block_size)

    count = 0
    for a in idx_h:
        for b in idx_w:
            img_block = x[:, :, a:a + block_size, b:b + block_size]
            batchs[count, :, :, :, ] = img_block
            count = count + 1
    batchs = batchs.to(device)
    return batchs, h, w

def de_batch(batchs, block_size, h, w, channels = 3):
    num_patches = batchs.size()[0]
    x = torch.zeros(1, channels, h, w)
    idx_h = range(0, h, block_size)
    idx_w = range(0, w, block_size)
    count = 0
    for a in idx_h:
        for b in idx_w:
            x[:, :, a:a + block_size, b:b + block_size] = batchs[count, :, :, :]
            count = count + 1
    return x