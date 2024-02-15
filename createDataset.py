import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import albumentations as A
import time
from tqdm import tqdm

# -- CHANGE 1 -- 
# Set the prefix for output files (can be modified)
output_prefix = "99948 (6114171)"
output_directory = "validation" # train, test, validation

# -- CHANGE 2 --
# Set the base directory and output subdirectories 
BASE_DIR = "piece_data"  # Set the base directory
OUTPUT_DIR_IMAGES = os.path.join('data', 'images', output_directory)  # Replace 'output_directory' with the directory where you want to save the image
OUTPUT_DIR_MASKS = os.path.join('data', 'masks', output_directory)
OUTPUT_DIR_LABELS = os.path.join('data', 'labels', output_directory)

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_DIR_MASKS, exist_ok=True)
os.makedirs(OUTPUT_DIR_LABELS, exist_ok=True)

obj_dict = {
    1: {'folder': "10928 (6012451)", 'longest_min': 150, 'longest_max': 800},
    2: {'folder': "13670 (6031821)", 'longest_min': 150, 'longest_max': 800},
    3: {'folder': "18575 (6346535)", 'longest_min': 150, 'longest_max': 800},
    4: {'folder': "2780 (4121715)", 'longest_min': 150, 'longest_max': 800},
    5: {'folder': "2815 (6028041)", 'longest_min': 150, 'longest_max': 800},
    6: {'folder': "29219 (6173127)", 'longest_min': 150, 'longest_max': 800},
    7: {'folder': "32009 (4495412)", 'longest_min': 150, 'longest_max': 800},
    8: {'folder': "32009 (6271161)", 'longest_min': 150, 'longest_max': 800},
    9: {'folder': "32013 (6284699)", 'longest_min': 150, 'longest_max': 800},
    10: {'folder': "32014 (6268905)", 'longest_min': 150, 'longest_max': 800},
    11: {'folder': "32034 (6271869)", 'longest_min': 150, 'longest_max': 800},
    12: {'folder': "32054 (4140806)", 'longest_min': 150, 'longest_max': 800},
    13: {'folder': "32062 (4142865)", 'longest_min': 150, 'longest_max': 800},
    14: {'folder': "32072 (6284188)", 'longest_min': 150, 'longest_max': 800},
    15: {'folder': "32073 (4211639)", 'longest_min': 150, 'longest_max': 800},
    16: {'folder': "32123 (4239601)", 'longest_min': 150, 'longest_max': 800},
    17: {'folder': "32140 (4141270)", 'longest_min': 150, 'longest_max': 800},
    18: {'folder': "32184 (4121667)", 'longest_min': 150, 'longest_max': 800},
    19: {'folder': "32270 (4177431)", 'longest_min': 150, 'longest_max': 800},
    20: {'folder': "32271 (6276836)", 'longest_min': 150, 'longest_max': 800},
    21: {'folder': "32278 (4542578)", 'longest_min': 150, 'longest_max': 800},
    22: {'folder': "32291 (6280394)", 'longest_min': 150, 'longest_max': 800},
    23: {'folder': "32316 (4211651)", 'longest_min': 150, 'longest_max': 800},
    24: {'folder': "32348 (4509912)", 'longest_min': 150, 'longest_max': 800},
    25: {'folder': "32348 (6278132)", 'longest_min': 150, 'longest_max': 800},
    26: {'folder': "32449 (4142236)", 'longest_min': 150, 'longest_max': 800},
    27: {'folder': "32498 (4177434)", 'longest_min': 150, 'longest_max': 800},
    28: {'folder': "32523 (4142822)", 'longest_min': 150, 'longest_max': 800},
    29: {'folder': "32523 (4153707)", 'longest_min': 150, 'longest_max': 800},
    30: {'folder': "32523 (4153718)", 'longest_min': 150, 'longest_max': 800},
    31: {'folder': "32523 (4509376)", 'longest_min': 150, 'longest_max': 800},
    32: {'folder': "32523 (6007973)", 'longest_min': 150, 'longest_max': 800},
    33: {'folder': "32524 (4495930)", 'longest_min': 150, 'longest_max': 800},
    34: {'folder': "32525 (4611705)", 'longest_min': 150, 'longest_max': 800},
    35: {'folder': "32526 (4211713)", 'longest_min': 150, 'longest_max': 800},
    36: {'folder': "32526 (4585040)", 'longest_min': 150, 'longest_max': 800},
    37: {'folder': "32556 (4514554)", 'longest_min': 150, 'longest_max': 800},
    38: {'folder': "32905 (6185471)", 'longest_min': 150, 'longest_max': 800},
    39: {'folder': "33299 (6313520)", 'longest_min': 150, 'longest_max': 800},
    40: {'folder': "3649 (6195314)", 'longest_min': 150, 'longest_max': 800},
    41: {'folder': "3673 (4211807)", 'longest_min': 150, 'longest_max': 800},
    42: {'folder': "3705 (370526)", 'longest_min': 150, 'longest_max': 800},
    43: {'folder': "3706 (370626)", 'longest_min': 150, 'longest_max': 800},
    44: {'folder': "3707 (370726)", 'longest_min': 150, 'longest_max': 800},
    45: {'folder': "3708 (370826)", 'longest_min': 150, 'longest_max': 800},
    46: {'folder': "3737 (373726)", 'longest_min': 150, 'longest_max': 800},
    47: {'folder': "40490 (4211866)", 'longest_min': 150, 'longest_max': 800},
    48: {'folder': "41239 (4522934)", 'longest_min': 150, 'longest_max': 800},
    49: {'folder': "41669 (6288218)", 'longest_min': 150, 'longest_max': 800},
    50: {'folder': "41678 (4162857)", 'longest_min': 150, 'longest_max': 800},
    51: {'folder': "4185 (4587275)", 'longest_min': 150, 'longest_max': 800},
    52: {'folder': "41897 (6035364)", 'longest_min': 150, 'longest_max': 800},
    53: {'folder': "42003 (6273715)", 'longest_min': 150, 'longest_max': 800},
    54: {'folder': "44294 (4211805)", 'longest_min': 150, 'longest_max': 800},
    55: {'folder': "44809 (6008527)", 'longest_min': 150, 'longest_max': 800},
    56: {'folder': "44874 (6321303)", 'longest_min': 150, 'longest_max': 800},
    57: {'folder': "4519 (4211815)", 'longest_min': 150, 'longest_max': 800},
    58: {'folder': "45590 (4198367)", 'longest_min': 150, 'longest_max': 800},
    59: {'folder': "4716 (4211510)", 'longest_min': 150, 'longest_max': 800},
    60: {'folder': "48989 (4225033)", 'longest_min': 150, 'longest_max': 800},
    61: {'folder': "49130 (6313453)", 'longest_min': 150, 'longest_max': 800},
    62: {'folder': "55013 (4499858)", 'longest_min': 150, 'longest_max': 800},
    63: {'folder': "56908 (4634091)", 'longest_min': 150, 'longest_max': 800},
    64: {'folder': "57518 (6325504)", 'longest_min': 150, 'longest_max': 800},
    65: {'folder': "57519 (4582792)", 'longest_min': 150, 'longest_max': 800},
    66: {'folder': "57585 (4502595)", 'longest_min': 150, 'longest_max': 800},
    67: {'folder': "59443 (4513174)", 'longest_min': 150, 'longest_max': 800},
    68: {'folder': "60483 (6265091)", 'longest_min': 150, 'longest_max': 800},
    69: {'folder': "60484 (4552347)", 'longest_min': 150, 'longest_max': 800},
    70: {'folder': "60485 (4535768)", 'longest_min': 150, 'longest_max': 800},
    71: {'folder': "63869 (6331441)", 'longest_min': 150, 'longest_max': 800},
    72: {'folder': "64178 (4540797)", 'longest_min': 150, 'longest_max': 800},
    73: {'folder': "64179 (4539880)", 'longest_min': 150, 'longest_max': 800},
    74: {'folder': "64392 (4541326)", 'longest_min': 150, 'longest_max': 800},
    75: {'folder': "64682 (4543490)", 'longest_min': 150, 'longest_max': 800},
    76: {'folder': "6536 (6261375)", 'longest_min': 150, 'longest_max': 800},
    77: {'folder': "6558 (4514553)", 'longest_min': 150, 'longest_max': 800},
    78: {'folder': "6562 (4666579)", 'longest_min': 150, 'longest_max': 800},
    79: {'folder': "6589 (4565452)", 'longest_min': 150, 'longest_max': 800},
    80: {'folder': "6590 (6275844)", 'longest_min': 150, 'longest_max': 800},
    81: {'folder': "6629 (6279881)", 'longest_min': 150, 'longest_max': 800},
    82: {'folder': "87080 (4566251)", 'longest_min': 150, 'longest_max': 800},
    83: {'folder': "87083 (6083620)", 'longest_min': 150, 'longest_max': 800},
    84: {'folder': "87086 (4566249)", 'longest_min': 150, 'longest_max': 800},
    85: {'folder': "92911 (6296844)", 'longest_min': 150, 'longest_max': 800},
    86: {'folder': "94925 (4640536)", 'longest_min': 150, 'longest_max': 800},
    87: {'folder': "99009 (4652235)", 'longest_min': 150, 'longest_max': 800},
    88: {'folder': "99010 (6326620)", 'longest_min': 150, 'longest_max': 800},
    89: {'folder': "99773 (6009019)", 'longest_min': 150, 'longest_max': 800},
    90: {'folder': "99948 (6114171)", 'longest_min': 150, 'longest_max': 800}
}

PATH_MAIN = "piece_data"

def get_img_and_mask(img_path, mask_path):

    img = cv2.imread(img_path)
    
    mask = cv2.imread(mask_path)
    
    mask_b = mask[:,:,0] == 0 # This is boolean mask
    mask = mask_b.astype(np.uint8) # This is binary mask
    
    return img, mask

def save_image_and_mask(image, mask, output_dir, idx):
    image_filename = os.path.join(output_dir, f"composition_{idx}.png")
    mask_filename = os.path.join(output_dir, f"composition_mask_{idx}.png")

    cv2.imwrite(image_filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(mask_filename, mask * 255)  # Assuming mask values are in [0, 1]

def resize_img(img, desired_max, desired_min=None):
   
    h, w = img.shape[0], img.shape[1]
    
    longest, shortest = max(h, w), min(h, w)
    longest_new = desired_max
    if desired_min:
        shortest_new = desired_min
    else:
        shortest_new = int(shortest * (longest_new / longest))
    
    if h > w:
        h_new, w_new = longest_new, shortest_new
    else:
        h_new, w_new = shortest_new, longest_new
        
    transform_resize = A.Compose([
        A.Sequential([
        A.Resize(h_new, w_new, interpolation=1, always_apply=False, p=1)
        ], p=1)
    ])

    transformed = transform_resize(image=img)
    img_r = transformed["image"]
        
    return img_r


def resize_transform_obj(img, mask, longest_min, longest_max, transforms=False):
   
    h, w = mask.shape[0], mask.shape[1]
    
    longest, shortest = max(h, w), min(h, w)
    longest_new = np.random.randint(longest_min, longest_max)
    shortest_new = int(shortest * (longest_new / longest))
    
    if h > w:
        h_new, w_new = longest_new, shortest_new
    else:
        h_new, w_new = shortest_new, longest_new
        
    transform_resize = A.Resize(h_new, w_new, interpolation=1, always_apply=False, p=1)

    transformed_resized = transform_resize(image=img, mask=mask)
    img_t = transformed_resized["image"]
    mask_t = transformed_resized["mask"]
        
    if transforms:
        transformed = transforms(image=img_t, mask=mask_t)
        img_t = transformed["image"]
        mask_t = transformed["mask"]
        
    return img_t, mask_t

transforms_bg_obj = A.Compose([
    A.RandomRotate90(p=1),
    A.ColorJitter(brightness=0.3,
                  contrast=0.3,
                  saturation=0.3,
                  hue=0.07,
                  always_apply=False,
                  p=1),
    A.Blur(blur_limit=(3,15),
           always_apply=False,
           p=0.5)
])

transforms_obj = A.Compose([
    A.RandomRotate90(p=1),
    A.RandomBrightnessContrast(brightness_limit=(0.0, 0.0),
                               contrast_limit=0.1,
                               brightness_by_max=True,
                               always_apply=False,
                               p=1)

])

def add_obj(img_comp, mask_comp, img, mask, x, y, idx):
    h_comp, w_comp = img_comp.shape[0], img_comp.shape[1]
    h, w = img.shape[0], img.shape[1]

    x = max(0, x - int(w / 2))  # Ensure x is not negative
    y = max(0, y - int(h / 2))  # Ensure y is not negative

    mask_b = mask == 1
    mask_rgb_b = np.stack([mask_b, mask_b, mask_b], axis=2)

    mask_added = None  # Default assignment

    if x < w_comp and y < h_comp:  # Check if the starting point is within the image boundaries
        h_part = min(h, h_comp - y)  # Corrected calculation for h_part
        w_part = min(w, w_comp - x)  # Corrected calculation for w_part

        if h_part > 0 and w_part > 0:  # Check if there is any overlap
            img_comp[y:y + h_part, x:x + w_part, :] = img_comp[y:y + h_part, x:x + w_part, :] * ~mask_rgb_b[:h_part, :w_part, :] + (img * mask_rgb_b)[:h_part, :w_part, :]
            mask_comp[y:y + h_part, x:x + w_part] = mask_comp[y:y + h_part, x:x + w_part] * ~mask_b[:h_part, :w_part] + (idx * mask_b)[:h_part, :w_part]
            mask_added = mask[:h_part, :w_part]

    return img_comp, mask_comp, mask_added

def create_bg_with_noise(files_bg_imgs,
                         files_bg_noise_imgs,
                         files_bg_noise_masks,
                         bg_max=1920,
                         bg_min=1080,
                         max_objs_to_add=60,
                         longest_bg_noise_max=1000,
                         longest_bg_noise_min=200,
                         blank_bg=False):
    
    if blank_bg:
        img_comp_bg = np.ones((bg_min, bg_max,3), dtype=np.uint8) * 255
        mask_comp_bg = np.zeros((bg_min, bg_max), dtype=np.uint8)
    else:    
        idx = np.random.randint(len(files_bg_imgs))
        img_bg = cv2.imread(files_bg_imgs[idx])
        img_bg = cv2.cvtColor(img_bg, cv2.COLOR_BGR2RGB)
        img_comp_bg = resize_img(img_bg, bg_max, bg_min)
        mask_comp_bg = np.zeros((img_comp_bg.shape[0], img_comp_bg.shape[1]), dtype=np.uint8)

    for i in range(1, np.random.randint(max_objs_to_add) + 2):

        idx = np.random.randint(len(files_bg_noise_imgs))
        img, mask = get_img_and_mask(files_bg_noise_imgs[idx], files_bg_noise_masks[idx])
        x, y = np.random.randint(img_comp_bg.shape[1]), np.random.randint(img_comp_bg.shape[0])
        img_t, mask_t = resize_transform_obj(img, mask, longest_bg_noise_min, longest_bg_noise_max, transforms=transforms_bg_obj)
        img_comp_bg, _, _ = add_obj(img_comp_bg, mask_comp_bg, img_t, mask_t, x, y, i)
        
    return img_comp_bg

def check_areas(mask_comp, obj_areas, overlap_degree=0.3):
    obj_ids = np.unique(mask_comp).astype(np.uint8)[1:-1]
    masks = mask_comp == obj_ids[:, None, None]
    
    ok = True
    
    if len(np.unique(mask_comp)) != np.max(mask_comp) + 1:
        ok = False
        return ok
    
    for idx, mask in enumerate(masks):
        if np.count_nonzero(mask) / obj_areas[idx] < 1 - overlap_degree:
            ok = False
            break
            
    return ok   


def create_composition(img_comp_bg,
                       max_objs=15,
                       overlap_degree=0.2,
                       max_attempts_per_obj=10):

    img_comp = img_comp_bg.copy()
    h, w = img_comp.shape[0], img_comp.shape[1]
    mask_comp = np.zeros((h,w), dtype=np.uint8)
    
    obj_areas = []
    labels_comp = []
    
    # Set the number of objects to 5 CHANGE FOR YOUR PURPOSE
    num_objs = 6
    
    i = 1
    
    # Generate a single random ratio for all objects
    ratio = np.random.uniform(0.5, 1.5)
    
    for _ in range(1, num_objs):

        obj_idx = np.random.randint(len(obj_dict)) + 1
        
        for _ in range(max_attempts_per_obj):

            imgs_number = len(obj_dict[obj_idx]['images'])
            idx = np.random.randint(imgs_number)
            img_path = obj_dict[obj_idx]['images'][idx]
            mask_path = obj_dict[obj_idx]['masks'][idx]
            img, mask = get_img_and_mask(img_path, mask_path)

            x, y = np.random.randint(w), np.random.randint(h)
            
            # Resize all objects using the same randomly generated ratio
            longest_min = int(obj_dict[obj_idx]['longest_min'] * ratio)
            longest_max = int(obj_dict[obj_idx]['longest_max'] * ratio)
            
            img, mask = resize_transform_obj(img,
                                             mask,
                                             longest_min,
                                             longest_max,
                                             transforms=transforms_obj)

            if i == 1:
                img_comp, mask_comp, mask_added = add_obj(img_comp,
                                                          mask_comp,
                                                          img,
                                                          mask,
                                                          x,
                                                          y,
                                                          i)
                obj_areas.append(np.count_nonzero(mask_added))
                labels_comp.append(obj_idx)
                i += 1
                break
            else:        
                img_comp_prev, mask_comp_prev = img_comp.copy(), mask_comp.copy()
                img_comp, mask_comp, mask_added = add_obj(img_comp,
                                                          mask_comp,
                                                          img,
                                                          mask,
                                                          x,
                                                          y,
                                                          i)
                ok = check_areas(mask_comp, obj_areas, overlap_degree)
                if ok:
                    obj_areas.append(np.count_nonzero(mask_added))
                    labels_comp.append(obj_idx)
                    i += 1
                    break
                else:
                    img_comp, mask_comp = img_comp_prev.copy(), mask_comp_prev.copy()        
        
    return img_comp, mask_comp, labels_comp, obj_areas



def create_yolo_annotations(mask_comp, labels_comp):
    comp_w, comp_h = mask_comp.shape[1], mask_comp.shape[0]

    obj_ids = np.unique(mask_comp).astype(np.uint8)[1:]
    masks = [mask_comp == obj_id for obj_id in obj_ids]

    annotations_yolo = []

    for i in range(len(labels_comp)):
        if i < len(masks):  # Check if the index is within the valid range
            mask = masks[i]
            pos = np.where(mask)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            xc = (xmin + xmax) / 2
            yc = (ymin + ymax) / 2
            w = xmax - xmin
            h = ymax - ymin

            annotations_yolo.append([labels_comp[i] - 1,
                                     round(xc/comp_w, 5),
                                     round(yc/comp_h, 5),
                                     round(w/comp_w, 5),
                                     round(h/comp_h, 5)])
        else:
            print(f"Skipping label {labels_comp[i]} due to index out of range.")

    return annotations_yolo

# Load in items
for k, v in obj_dict.items():
    folder_name = v['folder']
    
    files_imgs = sorted([f for f in os.listdir(os.path.join(PATH_MAIN, folder_name, 'images'))
                         if not f.startswith('.DS_Store')])
    files_imgs = [os.path.join(PATH_MAIN, folder_name, 'images', f) for f in files_imgs]
    
    files_masks = sorted([f for f in os.listdir(os.path.join(PATH_MAIN, folder_name, 'masks'))
                          if not f.startswith('.DS_Store')])
    files_masks = [os.path.join(PATH_MAIN, folder_name, 'masks', f) for f in files_masks]
    
    obj_dict[k]['images'] = files_imgs
    obj_dict[k]['masks'] = files_masks

#print("The first five files from the sorted list of 2780 images:", obj_dict[1]['images'][:5])
#print("\nThe first five files from the sorted list of 2780 masks:", obj_dict[1]['masks'][:5])

files_bg_imgs = sorted([f for f in os.listdir(os.path.join(PATH_MAIN, 'bg'))
                      if not f.startswith('.DS_Store')])
files_bg_imgs = [os.path.join(PATH_MAIN, 'bg', f) for f in files_bg_imgs]

files_bg_noise_imgs = sorted([f for f in os.listdir(os.path.join(PATH_MAIN, "bg_noise", "images"))
                             if not f.startswith('.DS_Store')])
files_bg_noise_imgs = [os.path.join(PATH_MAIN, "bg_noise", "images", f) for f in files_bg_noise_imgs]
files_bg_noise_masks = sorted([f for f in os.listdir(os.path.join(PATH_MAIN, "bg_noise", "masks"))
                              if not f.startswith('.DS_Store')])
files_bg_noise_masks = [os.path.join(PATH_MAIN, "bg_noise", "masks", f) for f in files_bg_noise_masks]

#print("\nThe first five files from the sorted list of background images:", files_bg_imgs[:5])
#print("\nThe first five files from the sorted list of background noise images:", files_bg_noise_imgs[:5])
#print("\nThe first five files from the sorted list of background noise masks:", files_bg_noise_masks[:5])

# -- CHANGE 3 --
# Number of iterations (scenes created) 
if output_directory == "train":
    num_iterations = 900 # (train 900) (validation 50) (test 50)
else:
    num_iterations = 50

# Choose the starting value for iteration_number
if output_directory == "train":
    start_iteration = 0 # (train 0) (validation 900) (test 950)
elif output_directory == "validation":
    start_iteration = 900
else:
    start_iteration = 950

# Loop for 5 iterations
for iteration_number in range(start_iteration, start_iteration + num_iterations):
    # Create composition
    img_comp_bg = create_bg_with_noise(files_bg_imgs,
                                       files_bg_noise_imgs,
                                       files_bg_noise_masks,
                                       max_objs_to_add=20)

    img_comp, mask_comp, labels_comp, obj_areas = create_composition(img_comp_bg,
                                                                     max_objs=15,
                                                                     overlap_degree=0.2,
                                                                     max_attempts_per_obj=10)

    # Get the index of the object specified by "output_prefix"
    output_obj_idx = None
    for idx, obj_info in obj_dict.items():
        if obj_info['folder'] == output_prefix:
            output_obj_idx = idx
            break

    if output_obj_idx is not None:
        # Ensure the specified object is placed after the first 5 random objects
        output_obj_path = obj_dict[output_obj_idx]['images'][0]
        output_mask_path = obj_dict[output_obj_idx]['masks'][0]
        output_img, output_mask = get_img_and_mask(output_obj_path, output_mask_path)
        x, y = np.random.randint(img_comp.shape[1]), np.random.randint(img_comp.shape[0])
        output_img, output_mask = resize_transform_obj(output_img, output_mask, 150, 800, transforms=transforms_obj)
        img_comp, mask_comp, _ = add_obj(img_comp, mask_comp, output_img, output_mask, x, y, 6)

        # Update output paths with the determined iteration number
        output_image_path = os.path.join(OUTPUT_DIR_IMAGES, f'{output_prefix}_{iteration_number:04d}.png')
        output_mask_path = os.path.join(OUTPUT_DIR_MASKS, f'{output_prefix}_{iteration_number:04d}_mask.png')
        output_labels_path = os.path.join(OUTPUT_DIR_LABELS, f'{output_prefix}_{iteration_number:04d}.txt')

        # Save images and annotations
        cv2.imwrite(output_image_path, img_comp)
        #cv2.imwrite(output_mask_path, mask_comp * 255) -- CHANGE uncomment this if you want to save masks

        # Include the information about the output object in the labels_comp list
        labels_comp.append(output_obj_idx)
        obj_areas.append(np.count_nonzero(output_mask))

        # Create YOLO annotations and save to a text file
        annotations_yolo = create_yolo_annotations(mask_comp, labels_comp)
        with open(output_labels_path, 'w') as f:
            for annotation in annotations_yolo:
                f.write(' '.join(str(el) for el in annotation) + '\n')

        print(f"Composite image saved at {output_image_path}")
        print(f"Composite mask saved at {output_mask_path}")
        print(f"YOLO annotations saved at {output_labels_path}")
    else:
        print(f"Object with output_prefix '{output_prefix}' not found in obj_dict.")