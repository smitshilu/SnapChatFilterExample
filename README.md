# SnapChatFilterExample

This is a simple SnapChat filter example using Computer Vision. 

Blog Post: [Medium](https://hackernoon.com/how-to-make-snapchat-lenses-f9eae861b5db)

Video: [YouTube](https://www.youtube.com/watch?v=pHBWW-sexf8)

## Demo
![Demo](https://github.com/smitshilu/SnapChatFilterExample/blob/master/sample.gif)

Required libraries:
  1. dlib
  2. opencv
  3. numpy
  
 ## Getting Stared
 ```
 # clone this repo
git clone https://github.com/smitshilu/SnapChatFilterExample.git
cd SnapChatFilterExample

# Start applying filters to video file and save it to another mp4 file
python apply_filter.py -f abc.mp4 -o xyz.mp4

# Start applying filters to video file and watch it without saving it
python apply_filter.py -f abc.mp4

# Start applying filters to camera feed
python apply_filter.py -f camera

# Start applying filters to camera feed and save it to another mp4 file
python apply_filter.py -f camera -o xyz.mp4

# By defalult it will take 30 FPS for any video but if you want to provide your own frame rate use following command
python apply_filter.py -f camera -o xyz.mp4 -fr 20
```

