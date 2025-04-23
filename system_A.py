#system_A.py
#homework 3 - Face Recognition "System A"
#Chamisa Edmo
#kuid: 2209458
#April 22, 2025

import cv2
import numpy as np
import pathlib
from tqdm import tqdm 

#pull given files into project 

#step 1 given function from homework write up
def normalizedCorrelationCoefficient(img1: np.ndarray, img2: np.ndarray):
    x = img1.reshape((-1,1))
    y = img2.reshape((-1,1))
    xn = x - np.mean(x)
    yn = y - np.mean(y)
    r = (np.sum(xn * yn)) / (np.sqrt(np.sum(xn**2)) * np.sqrt(np.sum(yn**2)))
    return r


gallery_file = pathlib.Path("GallerySet.rar")
probe_file = pathlib.Path("ProbeSet.rar")

#make sure imgs are named like "00023_01.jpg" and use text before first “_” as ID
def subject_id(filename: str) -> str:
    return filename.split("_")[0]

#load in the images and sort + strip them 
def load_folder(folder: pathlib.Path):
    imgs, ids = [], []
    for p in sorted(folder.glob("+")):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif"}:
            continue
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Cannot read this {p}")
        imgs.append(img.astype(np.float32)) #cast onee here
        ids.append(subject_id(p.name))
    return imgs, ids 

gallery_imgs, gallery_ids = load_folder(gallery_file)
probe_imgs, probe_ids = load_folder(probe_file)

assert len(gallery_imgs) == 100 and len(probe_imgs) == 100, "expect 100 gallery images and 100 probe images"

####
#resizing everything to the smallest common height and width, so vectors match 
h = min(img.shape[0] for img in gallery_imgs + probe_imgs)
w = min(img.shape[1] for img in gallery_imgs + probe_imgs)

N = len(probe_imgs)
M = len(gallery_imgs)
A = np.zeros((N, M), dtype=np.float32)

print("Computing ncc scores ...")
for i, pimg in enumerate(tqdm(probe_imgs, unit="probe")):
    for j, gimg in enumerate(gallery_imgs):
        A[i, j] = normalizedCorrelationCoefficient(pimg, gimg)

#required snippet
    print("\nA[0:10, 0:10] = ")
    with np.printoptions(precision=4, suppress=True):
        print(A[:10, :10])
