
![CookieBot Logo](https://github.com/DavimenUC3M/CookieClicker-Bot/blob/beaa33fd7e8fc9db8882b9dfdc70303089de2b61/art/CookieBot%20logo%20croped.png)


# Object detection YOLOv7 applied to cookie clicker game
YOLOv7 repository: https://github.com/WongKinYiu/yolov7

## Main features
* Autoclick on the big cookie 
* Autocollect all the golden cookies, red cookies, golden cartridges and red cartridges
* Autogarden function for planting gold clovers 
* Smaller model included for low end systems
* Possibility of run it over CPU or GPU

Currently the model has been trained only with 1080p and 1440p images, for other screen resolutions the bot might not work as expected

### Execution arguments
* **--use_tiny_model:** *Choose whether you want to use the tiny onnx model (Recommended for slower PCs)*
* **--run_type:** *Select if you want to run the NN on CUDA, TensorRT or CPU*
* **--real_time_window:** *If added, a pygame window will show the in real time object detector*
* **--real_time_window_width:** *Select the width of the in real time window*
* **--real_time_window_height:** *Select the height of the in real time window*
* **--toggle_key:** *Select the key you need to press in order to activate/deactivate the autoclick function*
* **--instagarden_toggle_key:** *Select the key you need to press in order to activate/deactivate the autoclick function*
* **--auto_garden:** *Choose whether you want or not the autogarden functionality*
* **--auto_garden_check:** *Select the time in seconds needed to check the garden*
* **--auto_garden_boost:** *Select the time in seconds added to the garden activation counter when using the slow compost*
* **--check_run_time_every:** *Select every how many minutes you want a print of the total run time*
* **--check_if_stuck_timer:** *Select every how many seconds you want to check if you are on the menu or options*

### How to install the required libraries
All the mandatory libraries are written in the conda_environment.yml file, to create the virtual conda environment type the following code on an Anaconda prompt: *conda env create -f conda_environment.yml*

## Trailer
[![Youtube trailer](https://img.youtube.com/vi/EBXYwDwHbGY/0.jpg)](https://youtu.be/EBXYwDwHbGY)

## Time lapse
[![Youtube time lapse](https://img.youtube.com/vi/gqcHEEmSMKM/0.jpg)](https://www.youtube.com/watch?v=gqcHEEmSMKM)

