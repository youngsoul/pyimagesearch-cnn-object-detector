# import the necessary packages
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from imutils.object_detection import non_max_suppression
from detection_helpers import sliding_window_generator
from detection_helpers import image_pyramid_generator
import numpy as np
import argparse
import imutils
import time
import cv2

# initialize variables used for the object detection procedure
WIDTH = 600
PYR_SCALE = 1.5
WIN_STEP = 16
ROI_SIZE = (200,150)
INPUT_SIZE = (224,224)

if __name__ == '__main__':

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to the input image")
    ap.add_argument("-s", "--window-size", type=str, default="(200, 150)",
                    help="ROI size (in pixels)")
    ap.add_argument("-c", "--min-conf", type=float, default=0.9,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-v", "--visualize", type=int, default=-1,
                    help="whether or not to show extra visualizations for debugging")
    ap.add_argument("--window-step", type=int, required=False, default=16, help="Sliding window step size. Default: 16.  Typical 4,8,16")
    args = vars(ap.parse_args())

    ROI_SIZE = eval(args['window_size'])
    WIN_STEP = args['window_step']


    # load our network weights from disk
    print("Loading iamge net network weights...")
    model = ResNet50(weights="imagenet", include_top=True)

    # load the input image from disk, resize it such that it has the supplied width
    # and then grab its dimensions
    orig_image = cv2.imread(args['image'])
    orig_image = imutils.resize(orig_image, width=WIDTH)
    (H,W) = orig_image.shape[:2]

    # initialize the image pyramid generator
    pyramid = image_pyramid_generator(orig_image, scale=PYR_SCALE, minSize=ROI_SIZE)

    # initialize two lists, one to hold the ROIs generated from the image
    # pyramid and sliding window, and another list used to store the
    # (x,y)-coordinates of where the ROI was in the original image
    rois = []
    locs = []

    # time how long it takes to loop over the image pyramid layers and sliding window locations
    start = time.time()

    # loop over the image pyramid
    for image in pyramid:
        # determine the scale factor between the original image dimensions and the current
        # layer of the pyramid
        # each image produced by the pyramid will have a progressively higher scale factor as it makes the
        # image smaller and smaller
        scale = W / float(image.shape[1])  # shape=(rows,columns,channels)=H,W,C

        # for each layer of the image pyramid, loop over the sliding window locations
        for (x,y,roiOrig) in sliding_window_generator(image, WIN_STEP, ROI_SIZE):
            # scale the (x,y)-coordinates of the ROI with respect to the original image dimensions
            x = int(x*scale)
            y = int(y*scale)
            w = int(ROI_SIZE[0] * scale)
            h = int(ROI_SIZE[1] * scale)

            # take the ROI and preprocess it so we can later classify
            # the region using Keras/Tensorflow
            roi = cv2.resize(roiOrig, INPUT_SIZE)
            roi = img_to_array(roi)
            roi = preprocess_input(roi)

            # update our list of ROIs and associated coordinates
            rois.append(roi)
            locs.append((x, y, x+w, y+h))

            # check to see if we are visualizing each of the sliding windows in the image pyramid
            if args['visualize'] > 0:
                # clone the original image and then draw a bounding box surrounding the current region
                clone = orig_image.copy()
                cv2.rectangle(clone, (x,y), (x+w, y+h), (0,255,0), 2)
                cv2.imshow("Visualization", clone)
                cv2.imshow("ROI", roiOrig)
                cv2.waitKey(0)

    # show how long it took to loop over the image pyramid layers and sliding window locations
    end = time.time()
    print(f"Looping over pyramid/windows took {(end-start):.5f} seconds")

    # convert the ROIs to a numpy array
    rois = np.array(rois, dtype='float32')

    # classify each of the proposal ROIs using ResNet and then show how long the classifications took
    print("Classifying ROIs...")
    start = time.time()
    preds = model.predict(rois)
    end = time.time()
    print(f"Classifying ROIs took {(end-start):.5f} seconds")

    # decode the predictions and intialize a dictionary which maps class labels (keys) to any ROIs
    # associated with that label ( values )
    preds = imagenet_utils.decode_predictions(preds, top=1)
    labels = {}

    # loop over the predictions
    for(i,p) in enumerate(preds):
        # grab the prediction information for the current ROI
        (imagenetID, label, prob) = p[0]

        # filter out weak detections by ensuring the predicted probability is greater than the minimum prob
        if prob >= args['min_conf']:
            # grab the bounding box associated with the prediction and convert the coordinates
            box = locs[i]

            # grab the list of predictions for hte label and add the bounding box and prob to the list
            L = labels.get(label, [])
            L.append((box, prob))
            labels[label] = L

    # loop over the labels for each of the detected objects in the image
    for label in labels.keys():
        # clone the original image so that we can draw on it
        print(f"Showing results for {label}")
        clone = orig_image.copy()

        # loop over all bounding boxes for the current label
        for (box, prob) in labels[label]:
            # draw the bounding box on the image
            (startX, startY, endX, endY) = box
            cv2.rectangle(clone, (startX, startY), (endX, endY), (0,255,0), 2)

        # show the results *before* applying non-maxima suppression, then
        # clone the image again so we can display the results *after*
        # applying non-maxima suppression
        cv2.imshow("Before NMS", clone)
        clone = orig_image.copy()

        # extract the bounding boxes and associated prediction
        # probabilities, then apply non-maxima suppression
        boxes = np.array([p[0] for p in labels[label]])
        proba = np.array([p[1] for p in labels[label]])
        boxes = non_max_suppression(boxes, proba)

        # loop over all bounding boxes that were kept *after* applying non-maxima suppression
        for (startX, startY, endX, endY) in boxes:
            # draw the bounding box and label on the image
            cv2.rectangle(clone, (startX, startY), (endX, endY), (0,255,0),2)
            text_y = startY-10 if startY-10 > 10 else startY+10
            cv2.putText(clone, label, (startX, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 2)

        # show the output after applying non-maxima suppression
        cv2.imshow("After NMS", clone)
        cv2.waitKey(0)
