import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


def main(path, filename):
    # Get the image and its initial shape
    img, init_shape = load_image(path+filename) # 70-100

    # Get greyscaled matrix and array and HSV of the image
    img_grey_mat, img_grey_arr, hsv = get_grey_and_hsv(img)

    # Get different channels of HSV
    hue, sat, val = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    sat_matrix = hsv[:,:,1]

    # Flatten the matrices
    hue.shape = (1,hue.shape[0]*hue.shape[1])
    sat.shape = (1,hue.shape[0]*hue.shape[1])

    # Calculate histogram
    hist = cv2.calcHist([img_grey_arr],[0],None,[256],[0,256])

    # Find local minimum to get rid of the background
    clip_min = local_minimum(hist)
    no_bg_mask = img_grey_arr < clip_min

    # Apply thresholding on the hue component
    # Get binary mask (0 and 255) for brown pixels
    ret,thresh_brown = cv2.threshold(hsv[:,:,0],20,255,cv2.THRESH_BINARY)
    thresh_brown = 255-thresh_brown
    # Get binary mask for brown pixels
    ret,thresh_blue = cv2.threshold(hsv[:,:,0],100,130,cv2.THRESH_BINARY)
    thresh_blue[:,:] = (thresh_blue[:,:]>0)*255

    # Get indeces for no background blue and brown regions
    no_bg_mask.shape = (init_shape[0], init_shape[1])
    index_brown = np.where(thresh_brown*no_bg_mask)
    #show_masks_and_histograms(no_bg_mask, img, [hist, hist])
    index_blue = np.where(thresh_blue*no_bg_mask)

    # Compute histograms
    hist_full_brown= cv2.calcHist([img_grey_mat[index_brown]],[0],None,[256],[0,256])
    hist_full_blue = cv2.calcHist([sat_matrix[index_blue]],[0],None,[256],[0,256])

    # Clip the brown histogram
    hist_brown_modified = hist_full_brown[:,0]
    hist_brown_modified = np.convolve(hist_brown_modified,[1,1,1,1,1,1,1,1,1,1]) # smooth
    max_loc1 = np.argmax(hist_brown_modified[0:150])
    max_loc2 = np.argmax(hist_brown_modified[151:255])+151
    clip_min = (max_loc1+max_loc2)/2

    mask_bright_brown = img_grey_mat[:,:]<clip_min
    index_brown_bright_not = np.where(mask_bright_brown*thresh_brown*no_bg_mask)
    browniest = np.zeros(init_shape)
    browniest[index_brown_bright_not] = img[index_brown_bright_not]

    # Clip the blue histogram
    hist_blue_modified = hist_full_blue[:,0]
    hist_blue_modified = np.convolve(hist_blue_modified,[1,1,1])# smooth
    max_loc = np.argmax(hist_blue_modified)
    clip_min = 2*max_loc

    mask_bright_blue = sat_matrix[:,:]>clip_min
    index_blue_bright_not = np.where(mask_bright_blue*thresh_blue*no_bg_mask)
    blueiest = np.zeros(init_shape)
    blueiest[index_blue_bright_not] = img[index_blue_bright_not]

    # Opening and Closing
    bin4morf = (blueiest>0.5)
    open_img = ndimage.binary_opening(bin4morf[:,:,1], structure=np.ones((10,10)))
    # Remove small black hole
    close_img_healthy = ndimage.binary_closing(open_img)

    # Opening and Closing
    bin4morf = (browniest>0.5)
    open_img = ndimage.binary_opening(bin4morf[:,:,1])
    # Remove small black hole
    close_img_cancer = ndimage.binary_closing(open_img)

    hue.shape = (init_shape[0], init_shape[1])
    sat.shape = (init_shape[0], init_shape[1])

    # Display original image with red contour around healthy and green contour around cancer cells
    save_segmented_result(img, close_img_healthy, close_img_cancer, filename)


# Load the sample image
def load_image(filename):
    img = cv2.imread(filename)
    print "read the image " + filename
    return img, img.shape


def get_grey_and_hsv(img):
    # Get grey for brightness evaluation (matrix and array)
    img_gray_mat = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img_gray_mat = cv2.equalizeHist(img_gray_mat)
    img_gray_arr = np.copy(img_gray_mat)
    img_gray_arr.shape = (1,img_gray_arr.shape[0]*img_gray_arr.shape[1])
    # Convert to hsv and get channels
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img_gray_mat, img_gray_arr, hsv


def local_minimum(hist):
    hit = hist[:,0]
    max_loc = np.argmax(hit)
    return 255-2*(255-max_loc)


def save_segmented_result(original_img, closed_heathly, closed_cancer, filename):
    # We load with opencv and plot with pyplot (BGR->RGB)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    plt.imshow(original_img)
    plt.contour(closed_heathly, [0.5], linewidths=0.5, colors='g')
    plt.contour(closed_cancer, [0.5], linewidths=0.5, colors='r')
    plt.savefig("../result/"+filename+"_output.jpg", dpi=1000)
    print(filename +" image saved.")


def save_result(image, filename):
    cv2.imwrite(filename, image)


if __name__ == '__main__':

    path = "../all/"

    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    for file in onlyfiles:
        main(path, file)
