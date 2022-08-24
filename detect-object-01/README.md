# 1. Introduction
learn-Yolo

# 2. Prerequisites
- make ```sudo apt install make```
- python3-dev python3-pip ```sudo apt install python3-dev python3-pip```
- Python >= 3.8

# 3. Document
[Mi AI guide](https://www.miai.vn/2019/08/05/yolo-series-1-su-dung-yolo-de-nhan-dang-doi-tuong-trong-anh)

# 4. Installation
- Install the requirements inside of a Python virtualenv (recommend)
```BASH
    pip install virtualenv
    virtualenv -p python3.8 venv
    source venv/bin/activate
```

- Make requirements
```BASH
    make requirements
```

- To deactivate
```
    deactivate
```

Download yolov3.weights file - weight của pretrain model from [Link](https://pjreddie.com/media/files/yolov3.weights)
to ./yolo folder

# 5. Commands
Run detect
```
python YOLO.py --image ./images/cam3.jpeg
```

# 6. Debug