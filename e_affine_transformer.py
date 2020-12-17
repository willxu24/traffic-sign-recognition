import os

import cv2
import numpy as np


def affine(img, delta_pix):
    rows, cols, _ = img.shape
    pts1 = np.float32([[0, 0], [rows, 0], [0, cols]])
    pts2 = pts1 + delta_pix
    M = cv2.getAffineTransform(pts1, pts2)
    res = cv2.warpAffine(img, M, (rows, cols))
    return res


def affine_dir(img_dir, write_dir, max_delta_pix):
    img_names = os.listdir(img_dir)
    # img_names = [img_name for img_name in img_names if img_name.split(".")[-1] == "png"]
    img_names = [img_name for img_name in img_names if img_name.split("_")[0] != "background"]
    for index, img_name in enumerate(img_names):
        img = cv2.imread(os.path.join(img_dir, img_name))
        save_name = os.path.join(write_dir, img_name.split(".")[0] + "f.png")
        delta_pix = np.float32(np.random.randint(-max_delta_pix, max_delta_pix + 1, [3, 2]))
        img_a = affine(img, delta_pix)
        cv2.imwrite(save_name, img_a)

        if index % 50 == 0:
            print("total image number = ", len(img_names), "current image number = ", index)


if __name__ == "__main__":
    img_dir = "./data/d_train_images"
    # write_dir = "./data/e_affine_images"
    affine_dir(img_dir, img_dir, 32)
    print("done")
