import reconstruction as rec
import argparse, os, sys, glob
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--testset",
        type=str,
        default="Set5",
        nargs="?",
        help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",
    )
    parser.add_argument(
        "--sr",
        type=str,
        default="1",
        nargs="?",
        help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="number of ddim sampling steps",
    )
    opt = parser.parse_args()
    setdir = "../datasets/"+opt.testset+"/"
    if opt.testset == "Set5" or opt.testset == "Set14":
        images = glob.glob(os.path.join(setdir, "*.bmp"))
    elif opt.testset == "BSD100":
        images = glob.glob(os.path.join(setdir, "*.jpg"))
    else:
        images = glob.glob(os.path.join(setdir, "*.png"))
    print(f"Found {len(images)} inputs.")

    config = "models/ldcsnet/"+opt.sr+"/config.yaml"
    ckptpath = "models/ldcsnet/"+opt.sr+"/model.ckpt"

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    outdir = "results/"+opt.testset+"/"+opt.sr+"/rec/"
    outoridir = "results/"+opt.testset+"/"+opt.sr+"/ori/"
    model = rec.get_model(config,ckptpath)

    os.makedirs(outdir, exist_ok=True)
    os.makedirs(outoridir, exist_ok=True)
    with torch.no_grad():
        with model["model"].ema_scope():
            for image in tqdm(zip(images)):
                outpath = os.path.join(outdir, os.path.split(image[0])[1])
                outpath = outpath.replace(".bmp",".png")
                outpath = outpath.replace(".jpg",".png")
                outpathori = outpath.replace("rec/","ori/")
                logs,orih,oriw,img_in = rec.run(model["model"],image[0],opt.steps)
                
                recon_x = logs["sample_noquant"].detach().cpu()
                recon_x = recon_x[:, :, 0:orih, 0:oriw]
                recon_x = torch.clamp((recon_x+1.0)/2.0, min=0.0, max=1.0)

                ori_image = img_in["image"].cpu()
                ori_image = ori_image[:, 0:orih, 0:oriw, :]
                ori_image = (ori_image+1.0)/2.0

                recon_x = recon_x.numpy().transpose(0,2,3,1)[0]*255
                ori_image = ori_image.cpu().numpy()[0]*255
                Image.fromarray(ori_image.astype(np.uint8)).save(outpathori)
                Image.fromarray(recon_x.astype(np.uint8)).save(outpath)
