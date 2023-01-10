import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    optional_args = parser._action_groups.pop()
    parser._action_groups.append(optional_args)

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

    parser.add_argument("-ac", "--auto_clicker",
                        action="store_true",
                        help="Choose whether you want or not the autoclick functionality at start." +
                             "Start with auto clicker by default")

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
                             "600 by default (10 mins)",
                        default=600)

    cliargs = parser.parse_args()
    return cliargs