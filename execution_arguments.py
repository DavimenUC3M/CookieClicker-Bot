import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    optional_args = parser._action_groups.pop()
    parser._action_groups.append(optional_args)

    parser.add_argument("-tm", "--use_tiny_model",
                        action="store_true",
                        help="Choose whether you want to use the tiny onnx model (Recommended for slower PCs)." +
                             "Not selected by default")

    parser.add_argument("-rt", "--run_type",
                        type=str,
                        help="Select if you want to run the NN on CUDA, TensorRT or CPU" +
                             "CUDA by default",
                        default="cuda")

    parser.add_argument("-rtw" ,"--real_time_window",
                        action="store_true",
                        help="If added, a pygame window will show the in real time object detector." +
                        "Showed by default")

    parser.add_argument("-rtww", "--real_time_window_width",
                        type=int,
                        help="Select the width of the in real time window." +
                             "1280 by default",
                        default=1280)

    parser.add_argument("-rtwh", "--real_time_window_height",
                        type=int,
                        help="Select the height of the in real time window." +
                             "720 by default",
                        default=720)

    parser.add_argument("-tk", "--toggle_key",
                        type=str,
                        help="Select the key you need to press in order to activate/deactivate the autoclick function." +
                             "f10 by default",
                        default="f10")

    parser.add_argument("-igtk", "--instagarden_toggle_key",
                        type=str,
                        help="Select the key you need to press in order to activate/deactivate the autoclick function." +
                             "f9 by default",
                        default="f9")

    parser.add_argument("-aim", "--auto_aim",
                        action="store_true",
                        help="Choose whether you want or not the autoaim functionality." +
                             "Active by default")

    parser.add_argument("-ag", "--auto_garden",
                        action="store_true",
                        help="Choose whether you want or not the autogarden functionality." +
                             "Not active by default")

    parser.add_argument("-agc", "--auto_garden_check",
                        type=int,
                        help="Select the time in seconds needed to check the garden" +
                             "300 by default (5 mins)",
                        default=300)

    parser.add_argument("-agb", "--auto_garden_boost",
                        type=int,
                        help="Select the time in seconds added to the garden activation counter when using the slow compost" +
                             "540 by default (9 mins)",
                        default=540)

    parser.add_argument("-crte", "--check_run_time_every",
                        type=int,
                        help="Select every how many minutes you want a print of the total run time" +
                             "Every 30 minutes by default",
                        default=30)

    parser.add_argument("-stuck", "--check_if_stuck_timer",
                        type=int,
                        help="Select every how many seconds you want to check if you are on the menu or options" +
                             "Every minute by default",
                        default=60)

    cliargs = parser.parse_args()
    return cliargs
