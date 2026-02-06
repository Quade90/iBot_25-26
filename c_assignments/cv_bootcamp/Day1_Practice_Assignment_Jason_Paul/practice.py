import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"OpenCV: {cv2.__version__}")
print("\n✅ All libraries loaded successfully!")

input()

print("x-------------x---- Section-1 ----x-------------x")

print("\nExercise 1.1: Array Creation\n")

array_2d = np.zeros((100, 150), dtype=np.uint8)

print(f"Shape: {array_2d.shape}")
print(f"Dimensions: {array_2d.ndim}")
print(f"Data Type: {array_2d.dtype}")
input()
print("-----\n")

print("Exercise 1.2: RGB Image Array\n")

array_color_2d = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)

print(f"Shape: {array_color_2d.shape}")
print(f"Total Elements: {array_color_2d.size}")
input()
print("-----\n")

print("Exercise 1.3: Array-Slicing - Center Crop\n")

array_color_2d = np.random.randint(0, 256, size=(200, 200, 3), dtype=np.uint8)

center_x = len(array_color_2d[0])//2
center_y = len(array_color_2d)//2

center_crop = array_color_2d[center_y - 50:center_y + 50, center_x - 50:center_x + 50]

print(f"Original image shape: {array_color_2d.shape}")
print(f"Center cropped image shape: {center_crop.shape}")
input()
print("-----\n")

print("Exercise 1.4: Boolean Indexing\n")

gray_2d = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)

pixel_count = np.sum(gray_2d > 127)
print(f"No. of pixels with pixel values > 127: {pixel_count}")

print(f"No. of pixels before at 255: {np.sum(gray_2d > 254)}")
gray_2d[gray_2d > 200] = 255
print(f"No. of pixels after at 255: {np.sum(gray_2d > 254)}")
input()
print("-----\n")

print(f"Exercise 1.5: Broadcasting - Adjusting RGB Channels\n")

rgb_img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
rgb_scale = np.array([1.2, 1.0, 0.9])
rgb_img_modified = rgb_scale * rgb_img
rgb_img_modified = np.clip(rgb_img_modified, a_min=0, a_max=255)

print(f"Original mean per channel: {rgb_img.mean(axis=(0,1))}")
print(f"Modified mean per channel: {rgb_img_modified.mean(axis=(0,1))}")
input()
print("-----\n")

print("Exercise 1.6: Reshape for deep learning\n")

single_image = np.random.rand(224, 224, 3)
single_image = np.expand_dims(single_image, axis=0)
print(f"Batched shape:{single_image.shape}")

single_image = np.transpose(single_image, (0, 3, 1, 2))
print(f"Channel-first shape:{single_image.shape}")
input()


print("x-------------x---- Section-2 ----x-------------x")

print("\nExercise 2.1: Create a Dataset Dataframe\n")
np.random.seed(42)
df = pd.DataFrame({
    'filename': [f'img_{i:03d}.jpg' for i in range(1, 11)],
    'label': np.random.choice(['cat', 'dog', 'bird'], 10),
    'width': np.random.randint(400, 1001, 10),
    'height': np.random.randint(300, 801, 10)
})  

print(df)
input()
print("-----\n")

print("Exercise 2.2: Filtering data\n")

df = pd.DataFrame({
    'filename': [f'img_{i:03d}.jpg' for i in range(1, 51)],
    'label': np.random.choice(['cat', 'dog', 'bird'], 50),
    'width': np.random.randint(400, 1001, 50),
    'height': np.random.randint(300, 801, 50)
})  

print(df, "\n")

df_dog = df[df['label'] == 'dog']
print(df_dog, "\n")

df_large_image = df[(df['width'] > 600) & (df['height'] > 500)]
print(df_large_image, "\n")

df_bird_cat = df[(df['label'] == 'cat') | (df['label'] == 'bird')]
print(df_bird_cat)
input()
print("-----\n")

print("Exercise 2.3: GroupBy Analysis\n")

sum = 0
avg_area = 0
class_large = ''
for i in df['label'].unique():
    print(f"No. of {i}s = {len(df[df['label'] == i])}")
    print(f"Average height = {df['height'].mean()}")
    print(f"Average width = {df['width'].mean()}\n")
    subset = df[df['label'] == i]
    avg_area = (subset['height'] * subset['width']).mean()
    if sum/len(df[df['label'] == i]) > avg_area:
        avg_area = sum/len(df[df['label'] == i])
        class_large = i

print(f"Class with largest average area: {i}\nArea: {avg_area}")
input()
print("-----\n")

print("Exercise 2.4: Train/Val/Test Split\n")

x = df[['width', 'height']]
y = df['label']

x_temp, x_test, y_temp, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

x_train, x_val, y_train, y_val = train_test_split(
    x_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

df['split'] = 'train'
df.loc[x_val.index, 'split'] = 'val'
df.loc[x_test.index, 'split'] = 'test'

print(f"Train: {len(x_train)} ({len(x_train)/len(df)*100:.0f}%)")
print(f"Val: {len(x_val)} ({len(x_val)/len(df)*100:.0f}%)")
print(f"Test: {len(x_test)} ({len(x_test)/len(df)*100:.0f}%)")

print("\nDistribution check:")
print(df.groupby(['split', 'label']).size().unstack(fill_value=0))
input()

print("x-------------x---- Section-3 ----x-------------x")

print("\nExercise 3.1: Create and Save a Synthetic Image\n")

blue_img = np.full((200, 200, 3), [255, 0, 0], dtype=np.uint8)
cv2.imwrite("blue_image.jpg", blue_img)

read_img = cv2.imread("blue_image.jpg")
plt.imshow(cv2.cvtColor(read_img, cv2.COLOR_BGR2RGB))
plt.title('Blue Image')
plt.axis('off')
plt.show()
print(f"Shape of blue image read: {read_img.shape}")
input()
print("-----\n")

print("Exercise 3.2: Color Space Conversions\n")

bgr_img = np.random.randint(0, 256, size=(200, 200, 3), dtype=np.uint8)
rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))

images = [bgr_img, rgb_img, gray_img, hsv_img]
titles = ["BGR", "RGB", "Grayscale", "HSV"]

for ax, img, title in zip(axes.flat, images, titles):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()
input()
print("-----\n")

print("Exercise 3.3: Resize with Aspect Ratio\n")

def resize_maintain_aspect(image, max_size):
    """Resize image keeping aspect ratio with longest side = max_size
    
    Parameters:
        image (numpy.ndarray): Input image
        max_size (int): Desired size of the longest side
    
    Returns:
        numpy.ndarray: Resized image"""

    h,w = image.shape[:2]
    if h > w:
        scale = max_size/h
    else:
        scale = max_size/w
    h = int(scale*h)
    w = int(scale*w)
    resized = cv2.resize(image, (w, h))

    return resized

test_img = np.random.randint(0, 256, (300, 500, 3), dtype=np.uint8)
test_img_resized = resize_maintain_aspect(test_img, 200)

print(f"Original image shape: {test_img.shape}")
print(f"Resized image shape: {test_img_resized.shape}")
input()
print("-----\n")

print("Exercise 3.4: Gaussian Blur Comparison\n")

image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
blur_3 = cv2.GaussianBlur(image, (3,3), sigmaX=0)
blur_9 = cv2.GaussianBlur(image, (9,9), sigmaX=0)
blur_21 = cv2.GaussianBlur(image, (21,21), sigmaX=0)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
images = [image, blur_3, blur_9, blur_21]
titles = ["No Blur", "Kernel: 3", "Kernel: 9", "Kernel: 21"]

for ax, img, title in zip(axes.flat, images, titles):
    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()
input()

print("x-------------x---- Section-4 ----x-------------x")

print("\nExercise 4.1: Image Statistics with Numpy\n")

image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)

mean_pixel = image.mean()
[mean_pixel_R, mean_pixel_G, mean_pixel_B] = image.mean(axis=(0,1))
print(f"Mean pixel values of all channels: {mean_pixel}")
print(f"Mean pixel values of red channel: {mean_pixel_R}")
print(f"Mean pixel values of green channel: {mean_pixel_G}")
print(f"Mean pixel values of blue channel: {mean_pixel_B}")

normalized_image = (image - image.min())/(image.max() - image.min())

standardized_image = (image - image.mean())/image.std()

print(f"Normalized mean: [{normalized_image.min():.4f},{normalized_image.max():.4f}")

print(f"Standardized image mean: {standardized_image.mean():.6f}")
print(f"Standardized image std: {standardized_image.std():.6f}")
input()
print("-----\n")

print("Exercise 4.2: Batch Processing Pipeline\n")

df = pd.DataFrame(columns=['filename', 'original_height', 'original_width'])

images = []

np.random.seed(42)

for i in range(1,6):
    image = np.random.randint(0, 256, size=(i*50, i*100, 3), dtype=np.uint8)
    images.append(image)
    img_info = pd.DataFrame([{
        'filename': f'img_{i:03d}.jpg',
        'original_height': f'{image.shape[0]}',
        'original_width': f'{image.shape[1]}'
    }])
    df = pd.concat([df, img_info], ignore_index=True)

print(df)

del img

for i, img in enumerate (images):
    img = cv2.resize(img, (224, 224))
    images[i] = img

batch = np.stack(images, axis=0)

print(f"Batch shape: {batch.shape}")

mean_pixel_values = {}

for i, img in enumerate(batch):
    pixel_mean = {f"img_{(i+1):03d}" : round(float(img.mean()), 6)}
    mean_pixel_values.update(pixel_mean)

print(f"Images and their mean pixel values: {mean_pixel_values}")
input()
print("-----\n")

print("Exercise 4.3: Simple Edge Detection Pipeline\n")

height, width = 180, 320
image = np.zeros((180, 320, 3), dtype=np.uint8)
pt1 = (width // 4, height // 4)
pt2 = (3*(width // 4), 3*(height // 4))
rect_img = cv2.rectangle(image, pt1, pt2, (255, 255, 255), 10)

gray = cv2.cvtColor(rect_img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (21, 21), sigmaX=0)
edges = cv2.Canny(blur, 100, 200)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
images = [image, edges]
titles = ["Original", "Detected Edges"]

for ax, img, title in zip(axes.flat, images, titles):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()
input()
print("-----\n")

print("Exercise 4.4: Image Brightness Analysis with Pandas\n")

gray_images = {}
for i in range(10):
    gray_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    gray_images.update({f'img_{i:03d}':gray_image})
df = pd.DataFrame(columns=['image_id', 'mean_brightness', 'min_val', 'max_val', 'std'])
for i in gray_images:
    gray_img_info = pd.DataFrame([{
        'image_id': f'{i}.jpg',
        'mean_brightness': gray_images[i].mean(),
        'min_val': gray_images[i].min(),
        'max_val': gray_images[i].max(),
        'std': gray_images[i].std()
    }])
    df = pd.concat([df, gray_img_info], ignore_index=True)

def classify_brightness(mean_val):
    if mean_val > 170:
        return 'bright'
    elif mean_val < 85:
        return 'dark'
    else:
        return 'medium'

df['category'] = df['mean_brightness'].apply(classify_brightness)
print(f"Images: \n{df}")

print(f"No. of images in each brightness category: \n{df['category'].value_counts()}")
input()

print("x-------------x---- Final_Boss ----x-------------x")

print("\nFinal Challenge: Complete preprocessing Function\n")
        
def preprocess_for_model(image, target_size=(224, 224)):
    """Preprocess an image for neural network input.
    
    Args:
        image: BGR image as numpy array
        target_size: (width, height) tuple
    
    Returns:
        Preprocessed image with shape (1, H, W, 3) and values in [0, 1]
    """
    image = cv2.resize(image, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    eps = 1e-8
    normalized_image = (image - image.min())/(image.max() - image.min())
    batched = np.expand_dims(normalized_image, axis=0)
    return batched


test_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
preprocessed_image = preprocess_for_model(test_image)
del fig, axes
fig, axes = plt.subplots(1, 2, figsize=(8,4))

axes[0].imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
axes[0].set_title('Test Image')

axes[1].imshow(preprocessed_image[0])
axes[1].set_title('Preprocessed Image')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()

print("\nx-------------x-- Assignment Completed --x-------------x")