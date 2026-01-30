import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"OpenCV: {cv2.__version__}")
print("\n✅ All libraries loaded successfully!")

print("\nx-------------x---- Section-1 ----x-------------x")

print("\nExercise 1.1: Array Creation")

array_2d = np.zeros((100, 150), dtype=np.uint8)

print(f"Shape: {array_2d.shape}")
print(f"Dimensions: {array_2d.ndim}")
print(f"Data Type: {array_2d.dtype}")

print("\nExercise 1.2: RGB Image Array")

array_color_2d = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)

print(f"Shape: {array_color_2d.shape}")
print(f"Total Elements: {array_color_2d.size}")

print("\nExercise 1.3: Array-Slicing - Center Crop")

array_color_2d = np.random.randint(0, 256, size=(200, 200, 3), dtype=np.uint8)

center_x = len(array_color_2d[0])//2
center_y = len(array_color_2d)//2

center_crop = array_color_2d[center_y - 50:center_y + 50, center_x - 50:center_x + 50]

print(f"Original image shape: {array_color_2d.shape}")
print(f"Center cropped image shape: {center_crop.shape}")

print("\nExercise 1.4: Boolean Indexing")

gray_2d = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)

pixel_count = np.sum(gray_2d > 127)
print(f"No. of pixels with pixel values > 127: {pixel_count}")

print(f"No. of pixels before at 255: {np.sum(gray_2d > 254)}")
gray_2d[gray_2d > 200] = 255
print(f"No. of pixels after at 255: {np.sum(gray_2d > 254)}")

print(f"Exercise 1.5: Broadcasting - Adjusting RGB Channels")

rgb_img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
rgb_scale = np.array([1.2, 1.0, 0.9])
rgb_img_modified = rgb_scale * rgb_img
rgb_img_modified = np.clip(rgb_img_modified, a_min=0, a_max=255)

print(f"Original mean per channel: {rgb_img.mean(axis=(0,1))}")
print(f"Modified mean per channel: {rgb_img_modified.mean(axis=(0,1))}")

print("\nExercise 1.6: Reshape for deep learning")

single_image = np.random.rand(224, 224, 3)
single_image = np.expand_dims(single_image, axis=0)
print(f"Batched shape:{single_image.shape}")

single_image = np.transpose(single_image, (0, 3, 1, 2))
print(f"Channel-first shape:{single_image.shape}")

print("\nx-------------x---- Section-2 ----x-------------x")

print("\nExercise 2.1: Create a Dataset Dataframe")
np.random.seed(42)
df = pd.DataFrame({
    'filename': [f'img_{i:03d}.jpg' for i in range(1, 11)],
    'label': np.random.choice(['cat', 'dog', 'bird'], 10),
    'width': np.random.randint(400, 1001, 10),
    'height': np.random.randint(300, 801, 10)
})  

print(df)

print("\nExercise 2.2: Filtering data")

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

print("\nExercise 2.3: GroupBy Analysis")

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

print("\nExercise 2.4: Train/Val/Test Split")

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

print("\nx-------------x---- Section-3 ----x-------------x")

print("\nExercise 3.1: Create and Save a Synthetic Image")

blue_img = np.full((200, 200, 3), [255, 0, 0], dtype=np.uint8)
cv2.imwrite("blue_image.jpg", blue_img)

read_img = cv2.imread("blue_image.jpg")
plt.imshow(cv2.cvtColor(read_img, cv2.COLOR_BGR2RGB))
plt.title('Blue Image')
plt.axis('off')
plt.show()
print(f"Shape of blue image read: {read_img.shape}")

print("\nExercise 3.2: Color Space Conversions")

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
