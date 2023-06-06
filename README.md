# eye_of_the_bird
This file shows the basic usage of our code.
The code is used to extract suspicious objects from a pair of Lidar point clouds (`.las` files).
## 1. Installations
To clone the repository, run the following command in your terminal:
```
git clone https://github.com/shilodaum/eye_of_the_bird.git
```

To install the required python packages, run the following command in the root directory of the project:
```
pip install -r requirements.txt
```
Additionally, the code uses the API of `CloudCompare`, which can be downloaded [here](https://www.danielgm.net/cc/release/). The path to the executable of `CloudCompare` has to be set in `subtract/pyCloudCompare.py`.
## 2. Usage
Our code consists of three main parts:
1. Object detection and extraction
2. Feature extraction
3. Training and evaluating
All three parts are used in the file `main.py`. The code is structured as follows:
### 2.1 Object detection and extraction
To detect and extract objects from a pair of point clouds, we must set the directory of the point clouds in `main.py` (by default in `data/scans`), as well as the coordinates of the bounding box in which the objects are located. make sure `CloudCompare` is installed and the path to the executable is set in `subtract/pyCloudCompare.py`.
The extracted objects are saved in the directory `data/objects`.
### 2.2 Feature extraction
To extract features from the extracted objects, we must set the directory of the objects in `main.py` (by default in `data/objects`). 
In order to label the objects, we must manually move the extracted objects into the directory `data/objects/<label>`, for example `data/objects/car`. 
The extracted features are saved in the directory `data/features`, the extracted features are saved as json files containing the features of all objects in the subdirectories of `data/objects`.
### 2.3 Training and evaluating
To train and evaluate the model, we must set the directory of the features in `main.py` (by default in `data/features`).
The model is trained using a random-forest classifier, and a random train-test split. The classification results are viewed using matplotlib, and common measures are printed out (e.g. accuracy).