#!/usr/bin/env python3
#system_A.py 
#HW-3 “System A”
#Chamisa Edmo
#KUID 2209458
#22 Apr 2025

import cv2, numpy as np, pathlib
from tqdm import tqdm

#file paths
ROOT = pathlib.Path(__file__).parent
GALLERY_DIR = ROOT / "GallerySet"
PROBE_DIR   = ROOT / "ProbeSet"

#NCC routine
def normalizedCorrelationCoefficient(img1: np.ndarray, img2: np.ndarray) -> float:
    x = img1.reshape((-1, 1))
    y = img2.reshape((-1, 1))
    xn = x - x.mean();   yn = y - y.mean()
    denom = np.sqrt((xn**2).sum()) * np.sqrt((yn**2).sum())
    return 0.0 if denom == 0 else float((xn * yn).sum() / denom)

#Image loader
VALID = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.pgm','.gif'}
def subject_id(name: str) -> str: return name.split("_")[0]

def load(folder: pathlib.Path):
    imgs, ids = [], []
    for p in folder.rglob("*"):
        if p.suffix.lower() not in VALID: continue
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:                   # corrupted file?
            print("⚠️  cannot read", p);  continue
        imgs.append(img.astype(np.float32))
        ids.append(subject_id(p.stem))
    return imgs, ids

#Load sets
gallery_imgs, gallery_ids = load(GALLERY_DIR)
probe_imgs,   probe_ids   = load(PROBE_DIR)
print(f"Loaded {len(gallery_imgs)} gallery and {len(probe_imgs)} probe images")

#Resize
h = min(im.shape[0] for im in gallery_imgs + probe_imgs)
w = min(im.shape[1] for im in gallery_imgs + probe_imgs)
resize = lambda im: cv2.resize(im,(w,h),cv2.INTER_AREA)
gallery_imgs = [resize(im) for im in gallery_imgs]
probe_imgs   = [resize(im) for im in probe_imgs]

#Build score matrix
A = np.zeros((len(probe_imgs), len(gallery_imgs)), np.float32)
for i, p in enumerate(tqdm(probe_imgs, unit='probe')):
    for j, g in enumerate(gallery_imgs):
        A[i, j] = normalizedCorrelationCoefficient(p, g)

#Question 1 snippet
print("\nA[0:10, 0:10] =")
with np.printoptions(precision=4, suppress=True):
    print(A[:10, :10])

#Questions 2 & 3
genuine_scores, impostor_scores = [], []
for i, pid in enumerate(probe_ids):
    for j, gid in enumerate(gallery_ids):
        (genuine_scores if pid == gid else impostor_scores).append(A[i, j])

genuine_scores  = np.array(genuine_scores,  np.float32)
impostor_scores = np.array(impostor_scores, np.float32)

lowest_genuine   = genuine_scores.min()
highest_impostor = impostor_scores.max()
print(f"\nLowest genuine score   : {lowest_genuine:.4f}")
print(f"Highest impostor score : {highest_impostor:.4f}")

mu1, mu0   = genuine_scores.mean(),  impostor_scores.mean()
std1, std0 = genuine_scores.std(ddof=1), impostor_scores.std(ddof=1)
d_prime = abs(mu1 - mu0) / np.sqrt(0.5 * (std1**2 + std0**2))
print(f"\nd′ (decidability index): {d_prime:.4f}")
