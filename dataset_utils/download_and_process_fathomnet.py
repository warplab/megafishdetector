from fathomnet.api import images as fm_imgs, boundingboxes as fm_bboxes
from fathomnet.api import taxa
from fathomnet.models import GeoImageConstraints
from urllib.request import urlretrieve

import cv2

import os
import sys
from os.path import join, isdir, exists
from tqdm import tqdm

OUTPUT_DIR = sys.argv[1]
OUTPUT_ROOT_NAME = "FATHOMNET_FISH"
DRY_RUN = False

def download_image(image_record, output_directory, overwrite=False):
    """
    Downloads image to output directory, overwrites only if specified
    """
    url = image_record.url
    extension = os.path.splitext(url)[-1]
    uuid = image_record.uuid
    image_filename = os.path.join(output_directory, image_record.uuid + extension)
    if not exists(image_filename) or overwrite:
        urlretrieve(url, image_filename)        # Download the image
    return image_filename

# Grab all vertebrates from MBARI taxanomy
#constraints = GeoImageConstraints(concept = "Vertebrata", taxaProviderName = 'mbari')
#all_vertebrate_images = fm_imgs.find(constraints)

# Grab all fish-like things
# TODO: filter by verified as well...
# TODO: pickle this array so we don't have to keep querying server
print("Fetching all image records from mbari:Gnathostomata")
constraints = GeoImageConstraints(concept = "Gnathostomata", taxaProviderName = 'mbari') #"Delphinidae", for testing
all_fish_images = fm_imgs.find(constraints)
whitelisted_taxa = taxa.find_taxa('mbari', 'Gnathostomata')
whitelisted_concepts = set([t.name for t in whitelisted_taxa])

# Grab all concepts as blacklist, and remove the ones we want
# TODO: implement this (should probably cache all these names into a file, and come up with a faster way to use them too)
blacklisted_taxa = taxa.find_taxa('mbari', 'physical object')
blacklisted_concepts = set([t.name for t in blacklisted_taxa])

image_records = []
image_records.extend(all_fish_images)

if not isdir(join(OUTPUT_DIR, "images/training")):
    os.makedirs(join(OUTPUT_DIR, "images/training"), exist_ok=True)

if not isdir(join(OUTPUT_DIR, "labels/training")):
    os.makedirs(join(OUTPUT_DIR, "labels/training"), exist_ok=True)

if not isdir(join(OUTPUT_DIR, "images/validation")):
    os.makedirs(join(OUTPUT_DIR, "images/validation"), exist_ok=True)

if not isdir(join(OUTPUT_DIR, "labels/validation")):
    os.makedirs(join(OUTPUT_DIR, "labels/validation"), exist_ok=True)

if not isdir(join(OUTPUT_DIR, "images/test")):
    os.makedirs(join(OUTPUT_DIR, "images/test"), exist_ok=True)

if not isdir(join(OUTPUT_DIR, "labels/test")):
    os.makedirs(join(OUTPUT_DIR, "labels/test"), exist_ok=True)

if not isdir(join(OUTPUT_DIR, "images/groundtruth")):
    os.makedirs(join(OUTPUT_DIR, "images/groundtruth"), exist_ok=True)

with open(join(OUTPUT_DIR, "dataset.yaml"), "w") as f_out:
    f_out.write(f"path: ../datasets/{OUTPUT_ROOT_NAME}\n")
    f_out.write("train: images/training\n")
    f_out.write("val: images/validation\n")
    f_out.write("test: images/test\n")
    f_out.write("nc: 1\n")
    f_out.write("names: ['fish']")

num_total_images = len(image_records)
image_counter = 0
num_total_bboxes = 0
num_verified_bboxes = 0
num_included_bboxes = 0
num_excluded_bboxes = 0

print(f"Downloading and processing images: {num_total_images} expected images")

if DRY_RUN:
    print("DRY RUN")

    for img_record in tqdm(image_records):
        for bbox in img_record.boundingBoxes:
            num_total_bboxes += 1

            if bbox.verified:
                num_verified_bboxes += 1
                
                if not bbox.concept in whitelisted_concepts:
                    num_excluded_bboxes += 1
                    #print(f"Excluding bbox for {bbox.concept}")
                    continue
                num_included_bboxes += 1
        image_counter+=1

    print(f"Num total images: {num_total_images}")
    print(f"Num total included bboxes: {num_included_bboxes}")
    print(f"Num total excluded bboxes: {num_excluded_bboxes}")
    print(f"Num total bboxes: {num_total_bboxes}")
    print(f"Num verified bboxes: {num_verified_bboxes}")
    
    exit()
    
for img_record in tqdm(image_records):

    #print(f"Processing: uuid-{img_record.uuid}, url-{img_record.url}, image #-{image_counter}")
    img_filename = download_image(img_record, join(OUTPUT_DIR, "images/training"))
    img_basename = os.path.basename(img_filename)
    img_name = os.path.splitext(img_basename)[0]
    
    img = cv2.imread(img_filename)
    img_bboxes = img.copy()
    img_w = img_record.width
    img_h = img_record.height

    with open(join(OUTPUT_DIR, "labels/training", img_name + ".txt"), "w") as f_out:
    
        for bbox in img_record.boundingBoxes:
            num_total_bboxes += 1

            if bbox.verified:
                num_verified_bboxes += 1

            if not bbox.concept in whitelisted_concepts:
                num_excluded_bboxes += 1
                #print(f"Excluding bbox for {bbox.concept}")
                continue

            num_included_bboxes += 1
            
            x = bbox.x
            y = bbox.y
            w = bbox.width
            h = bbox.height
            
            yolo_x = ((int(x)+int(w)/2)) / int(img_w)
            yolo_y = ((int(y)+int(h)/2)) / int(img_h)
            yolo_w = int(w) / int(img_w)
            yolo_h = int(h) / int(img_h)

            f_out.write(f"0 {yolo_x} {yolo_y} {yolo_w} {yolo_h}\n")
            img_bboxes = cv2.rectangle(img_bboxes, (x, y), (x+w, y+h), (0, 0, 255))

        cv2.imwrite(join(OUTPUT_DIR, "images/groundtruth", img_basename), img_bboxes)

    image_counter += 1

print(f"Num total images: {num_total_images}")
print(f"Num total included bboxes: {num_included_bboxes}")
print(f"Num total excluded bboxes: {num_excluded_bboxes}")
print(f"Num total bboxes: {num_total_bboxes}")
print(f"Num verified bboxes: {num_verified_bboxes}")


