# SnapChatFilterExample

This is a simple SnapChat filter example using Computer Vision. 

Required libraries:
  1. dlib
  2. opencv
  3. numpy
  
 ## Getting Stared
 ```
 # clone this repo
git clone https://github.com/smitshilu/SnapChatFilterExample.git
cd SnapChatFilterExample
# Start applying filters
python pix2pix.py \
  --mode test \
  --output_dir facades_test \
  --input_dir facades/val \
  --checkpoint facades_train
  ```
