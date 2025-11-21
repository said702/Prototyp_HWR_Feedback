import os
import lmdb
import cv2
import numpy as np


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

def createDataset(gtFile,checkValid=True):
    inputPath = "extracted_bboxes_for_prediction"  
    outputPath = "lmdb_dataset" 

    """
    Create LMDB dataset for evaluation when no labels are provided.
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=10 * 1024 * 1024)
    cache = {}
    cnt = 1

    with open(gtFile, 'r', encoding='utf-8') as data:
        datalist = data.readlines()

    nSamples = len(datalist)
    for i in range(nSamples):
        try:
            imagePath, label = datalist[i].strip('\n').split('\t')
        except ValueError:
            # Default to empty label if none provided
            imagePath, label = datalist[i].strip('\n'), ""

        imagePath = os.path.join(inputPath, imagePath)

        if not os.path.exists(imagePath):
            print(f"{imagePath} does not exist")
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print(f"{imagePath} is not a valid image")
                    continue
            except Exception as e:
                print(f"Error occurred for {imagePath}: {e}")
                continue

        imageKey = f"image-{cnt:09d}".encode()
        labelKey = f"label-{cnt:09d}".encode()
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print(f"Written {cnt} / {nSamples}")
        cnt += 1

    nSamples = cnt - 1
    cache["num-samples".encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print(f"Created dataset with {nSamples} samples")

def generate_gt_file():
    input_path = "extracted_bboxes_for_prediction"  
    """
    Generate a gt.txt file with empty labels for images in the input directory.
    """
    gt_file_path = os.path.join(input_path, "gt.txt")

    # Numerische Sortierung der Dateinamen
    image_names = sorted(
        [img for img in os.listdir(input_path) if img.endswith((".jpg", ".png"))],
        key=lambda x: int(x.split("_")[1].split(".")[0])  # Extrahiere die Nummer
    )

    # Schreiben der sortierten Namen in die Datei
    with open(gt_file_path, "w") as f:
        for image_name in image_names:
            f.write(f"{image_name}\t\n")  # Image name with empty label

    return gt_file_path


