# Instance-Level Object Detection

This repository implements an **instance-level object detection** system developed as part of the TTIC 31040 Computer Vision course project.

## 🔍 Project Description

Given a **reference image** of an object and a **test image**, this system identifies all instances of the object in the test image. The reference image contains only the object of interest, and the object is not occluded.

The algorithm uses **SIFT (Scale-Invariant Feature Transform)** keypoints to detect and match features between the reference and test images. It then estimates geometric transformations to locate multiple instances in the scene.

### Outputs

- ✅ The test image annotated with bounding boxes around detected instances.
- 📍 The (x, y) coordinates of the bounding box corners.
- 🔢 The total number of detected instances.

## 📁 Project Structure

- `visualize.ipynb`: Main notebook that runs the instance detection pipeline.
- `helper_functions.py`: Core implementation of feature detection and matching.
- `requirements.txt`: Python dependencies for the project.
- `Sounak-Paul-FinalReport.pdf`: Full project report with explanation and results.

## ⚙️ Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/paulsounak96/instance-level-object-detect.git
cd instance-level-object-detect
```

2. **Create and activate a virtual environment**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **(Optional) Install Jupyter**
```bash
pip install jupyter
```

## ▶️ Usage Instructions

1. Launch Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `visualize.ipynb` and run all cells.

3. The notebook will:
   - Load the reference and test images.
   - Use SIFT to find matches.
   - Detect all instances of the object in the test image.
   - Display the annotated test image with bounding boxes and print instance data.

## 📚 Citation

This project is based on the SIFT feature descriptor introduced by David Lowe. Please cite the following paper if you use this work:

> David G. Lowe. *"Distinctive Image Features from Scale-Invariant Keypoints"*, International Journal of Computer Vision, 60, 91–110 (2004).  
> [Springer Link](https://link.springer.com/article/10.1023/B:VISI.0000029664.99615.94)

## 🪪 License

This project is licensed under the MIT License.
