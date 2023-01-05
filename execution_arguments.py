import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    optional_args = parser._action_groups.pop()
    parser._action_groups.append(optional_args)

    parser.add_argument("-rtv" ,"--real_time_viewer",
                        action="store_true",
                        help="If added, a pygame window will show the in real time object detector." +
                        "Showed by default")

    parser.add_argument("-rtvw", "--real_time_viewer_width",
                        type=int,
                        help="Select the width of the in real time window." +
                             "1280 by default",
                        default=1280)

    parser.add_argument("-rtvh", "--real_time_viewer_height",
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
                             "0 by default",
                        default="f10")

    parser.add_argument("-aim", "--auto_aim",
                        action="store_true",
                        help="Choose whether you want or not the autoaim functionality." +
                             "Active by default")

    cliargs = parser.parse_args()
    return cliargs