# Image Classification Comparison

## Dataset
- **Training data**: 100 images in per class (15 classes, 1500 images total)
- **Test data**: 2985 unclassified images

## Methods

### Method 1: k-Nearest Neighbour (Tiny Image Features)
- Uses **k-nearest neighbour (k-NN)** as the classifier
- Feature extraction:
  1. Crop image to a central square
  2. Resize to **16x16 pixels**
  3. Flatten pixels into a vector
  4. Normalise: **zero mean, unit length**

### Method 2: Linear Classifiers (Bag-of-Visual-Words)
- Uses an ensemble of **15 one-vs-all linear classifiers**
- Feature extraction:
  1. **Densely sample 8x8 pixel patches** (every 4 pixels in x and y directions)
  2. Mean-centre and normalise before clustering
  3. Use **K-means clustering** to form a vocabulary of visual words
  4. Use **Vector quantisation** to map patches to visual words

### Method 3: Support Vector Machine (GIST Descriptors)
- Uses a **Support Vector Machine (SVM)** with a **Radial Basis Function (RBF) kernel**
- Feature extraction:
  1. Create a **filter bank of 32 Gabor filters** (8 orientations x 4 scales)
  2. Convert the images to **grayscale**
  3. Apply a **Gaussian filter**
  4. Resize to **256x256 pixels** while maintaining the original aspect ratio
  5. Apply the **filter bank** to each image
  6. Divide each image into a **4x4** grid, and calculate the mean of each grid cell
  7. Each cell mean represents a feature, concatenated to form the GIST descriptor
  8. Given **32 filters** and **16 cells per image**, each descriptor contains **512 features**
- **Multiprocessing** was used to speed up feature extraction

## Getting Started

### Installation
1. **Clone the repository**
    ```bash
    git clone https://github.com/simran-ss-sandhu/Image-Classification-Comparison.git
    ```
2. **Navigate to the project directory**
    ```bash
    cd Image-Classification-Comparison
    ```
3. **Set up a virtual environment (optional but recommended)**
    - **Windows**
        ```bash
        python -m venv venv && venv\Scripts\activate
        ```
    - **UNIX (Linux, MacOS, etc.)**
        ```bash
        python -m venv venv && source venv/bin/activate
        ```
4. **Install dependencies**
    ```bash
    pip install -e .
    ```

### Usage
Start the project by using the following command in the project directory:
```bash
python -m image_classification_comparison
```

## Results (average precision)
- [Method 1](#method-1-k-nearest-neighbour-tiny-image-features): 25.6%
- [Method 2](#method-2-linear-classifiers-bag-of-visual-words): 61.3%
- [Method 3](#method-3-support-vector-machine-gist-descriptors): 82.4%

## Authors
- Simran Sandhu
- Henry Card
- James Martin
- Ayush Varshney

## Acknowledgements
- [A.Oliva and A.Torralba, Modelling the shape of the scene: a holistic representation of the spatial envelope, 2001](http://people.csail.mit.edu/torralba/code/spatialenvelope/)
