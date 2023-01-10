import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def template_matching(img, template="Big_cookie", resolution=1440, threshold=-1, RGB=True, verbose=False):
    # img: Image to perform the template matching.
    # template: Selected template image.
    # resolution: resolution of the game (currently only 1440p is supported)
    # threshold: If float between 0 and 1 return the center coords of all the object detected with the selected or greater similarity level,
    #            if other number, return the center coords of the object with the maximum similarity value.
    # RGB: If true transforms the image to the BGR format which is the preferred one for opencv library.
    # verbose: If true prints the obtained center coords along with a plot of the image with the bounding boxes of the detected items.

    # returns the central screen coordinates of the detected objects

    img = img.copy() # Creating a copy of the image
    img_with_bx = img.copy() # image to add bounding boxes on top

    holes_dirs = ["empty"] # Initializing the variable in case Holes template is not selected

    # Selecting the template
    if resolution == 1080:
        template_path = 'Template_Matching_Imgs/1080p/'
    else:
        template_path = 'Template_Matching_Imgs/1440p/'

    if template == "Big_cookie":
        template = cv2.imread(template_path + "Big_cookie.png")
    elif template == "Silent_parcel":
        template = cv2.imread(template_path + "Silent_parcel.png")
    elif template == "Open_garden":
        template = cv2.imread(template_path + "Open_garden.png")
    elif template == "Close_garden":
        template = cv2.imread(template_path + "Close_garden.png")
    elif template == "Farm_icon":
        template = cv2.imread(template_path + "Farm_icon.png")
    elif template == "Crop_remover":
        template = cv2.imread(template_path + "crop_remover.png")
    elif template == "Target_seed":
        template = cv2.imread(template_path + "target_seed.png")
    elif template == "Target_plant1":
        template = cv2.imread(template_path + "target_plant1.png")
    elif template == "Target_plant2":
        template = cv2.imread(template_path + "target_plant2.png")
    elif template == "Target_plant3":
        template = cv2.imread(template_path + "target_plant3.png")
    elif template == "Target_plant4":
        template = cv2.imread(template_path + "target_plant4.png")
    elif template == "Target_plant5":
        template = cv2.imread(template_path + "target_plant5.png") # At least, in the case of the golden trebol, plant4 and plant5 are both equivalent to this algorithm
    elif template == "Speed_compost":
        template = cv2.imread(template_path + "Speed_compost.png")
    elif template == "Slow_compost":
        template = cv2.imread(template_path + "Slow_compost.png")
    elif template == "Holes":
        template_path += "Garden_holes"
        holes_dirs = os.listdir(template_path) # This case is special and has its own pipeline
        total_holes = len(holes_dirs)

    else:
        template = template.copy() # If the input is an image keep it as it is

    if RGB:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_with_bx = cv2.cvtColor(img_with_bx, cv2.COLOR_RGB2BGR)

    coords = []
    for hole in holes_dirs:

        if hole != "empty":
            template = cv2.imread(template_path + "/" + hole)


        # read height and width of template image
        w, h = template.shape[1], template.shape[0]

        template_match = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

        if threshold >= 0 and threshold <= 1:
            loc = np.where(template_match >= threshold)

            center_coords = []
            for pt in zip(*loc[::-1]):
                cv2.rectangle(img_with_bx, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
                coord = ((pt[0] + pt[0] + w) // 2, (pt[1] + pt[1] + h) // 2)
                if verbose:
                    print("Center coords: ", coord, "Similarity:", template_match[pt[1], pt[0]])
                center_coords.append(coord)
                coords.append(center_coords)  # Only useful when looping through the holes templates
            center_coords.reverse()

            #if len(center_coords) == 1:
                #center_coords = center_coords[0]  # Unleashing the list

        else:
            pt = list(zip(*np.where(template_match == np.amax(template_match))[::-1]))[0]
            if verbose:
                print("Max similarity value:", np.amax(template_match))
            cv2.rectangle(img_with_bx, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)

            center_coords = ((pt[0] + pt[0] + w) // 2, (pt[1] + pt[1] + h) // 2)

            coords.append(center_coords) # Only useful when looping through the holes templates

            if verbose:
                print("Center coords: ", center_coords)

    if verbose:
        img_with_bx = cv2.cvtColor(img_with_bx, cv2.COLOR_BGR2RGB)
        plt.imshow(img_with_bx)


    if hole != "empty":
        coords = [x for x in coords if x] # Removing empty lists
        return coords

    return center_coords
