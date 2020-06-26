import imutils


def sliding_window_generator(image, step, ws):
    """


    :param image: shape=(rows,columns,channels) shape[0] = height, shape[1] = width, shape[2]= # color channels
    :type image:
    :param step: step size, the number of pixels to skip in both x,y directions.
    :type step:
    :param ws: window size, width/height in pixels, of window we are going to extract from our image
                ws[0] = width
                ws[1] = height
    :type ws:
    :return:
    :rtype:
    """

    # shape[0] - ws[1] = image height - window height = y space not covered by the window
    for y in range(0, image.shape[0] - ws[1], step):
        # shape[1] - ws[0] = image width - window width = x space not covered by the window
        for x in range(0, image.shape[1] - ws[0], step):
            # yield the current window
            yield (x, y, image[y:y + ws[1], x:x + ws[0]])


def image_pyramid_generator(image, scale=1.5, minSize=(224, 224)):
    # yield the original image
    yield image

    # keep looping over the image pyramid
    while True:
        # compute the dimensions of the next image in the pyramid
        w = int(image.shape[1] / scale)  # scale the width for the scale factor
        image = imutils.resize(image, width=w)  # rescale the image for the calculated width
                                                # maintain aspect ratio

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        # yield the next image (resized) in the pyramid
        yield image
