# Prompt-Vision-Detection

A simple repository for experimenting with prompt-based vision detection models.  
This project contains training, testing, preprocessing, and visualization scripts, along with example data and pretrained weights.

## 📌 Branches

This repository includes the following branches:

- **master** – Main branch with core code and default development history. :contentReference[oaicite:0]{index=0}  
- **naflex** – Experimental branch for additional features and NaFLEX-related code. :contentReference[oaicite:1]{index=1}  
- **openvocab** – Experimental branch focused on open-vocabulary detection code and experiments. :contentReference[oaicite:2]{index=2}

You can switch to any branch using:

```bash
git checkout <branch_name>
📂 Directory Overview
This repo contains:

.gitignore
decoder_epoch10.pth
image.jpg
overlay_image.jpg
preprocess.py
test.py
train.py
visualize_dataset.py
weights_10k_kindagood.pth
weights_first_working.pth
weights_siglip_localizer.pth
weights_zero_object_detection.pth
(Core code files and pretrained weights) 

🚀 Getting Started
Clone the repository:

git clone https://github.com/albertbastl/Prompt-Vision-Detection.git
cd Prompt-Vision-Detection
Install dependencies:
(Add your dependencies in a requirements.txt file if missing)

pip install -r requirements.txt
Select a branch:

git fetch
git checkout openvocab
# or
git checkout naflex
Training:

python train.py
Testing:

python test.py --model path/to/weights.pth
🛠️ Scripts
train.py – Train the model from scratch or resume training. 

test.py – Evaluate a trained model. 

preprocess.py – Dataset preprocessing utilities. 

visualize_dataset.py – Utility to visualize dataset samples and annotations. 

📊 Example Images
image.jpg – Example input image. 

overlay_image.jpg – Overlay visualization example. 

📦 Weights
Included pretrained weights:

decoder_epoch10.pth

weights_10k_kindagood.pth

weights_first_working.pth

weights_siglip_localizer.pth

weights_zero_object_detection.pth
(All found in the repo) 

📝 Notes
This is an experimental repo with multiple feature branches. 

Documentation and setup instructions should be updated as features evolve.
