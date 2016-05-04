import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


def main():
    # Get the image and its initial shape
    img, init_shape = load_image('../all/16_36_3_3_768.jpg') # 70-100

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
    bin_img = img_grey_arr < clip_min
    indexes = np.where(bin_img)

    # Collect the non-background pixel values in different channels
    img_no_bg = img_grey_arr[indexes]
    hue_no_bg = hue[indexes]
    sat_no_bg = sat[indexes]

    hist_full_gray = cv2.calcHist([np.array(img_no_bg)],[0],None,[256],[0,256])
    hist_full_hue = cv2.calcHist([np.array(hue_no_bg)],[0],None,[256],[0,256])
    hist_full_sat = cv2.calcHist([np.array(sat_no_bg)],[0],None,[256],[0,256])

    # Apply thresholding on the hue component
    ret,thresh1 = cv2.threshold(hsv[:,:,0],20,255,cv2.THRESH_BINARY)
    ret,thresh2 = cv2.threshold(hsv[:,:,0],100,130,cv2.THRESH_BINARY)

    bin_img.shape = (init_shape[0], init_shape[1])
    thresh1 = 255-thresh1
    thresh2[:,:] = (thresh2[:,:]>0)*255
    index_brown = np.where(thresh1*bin_img)
    index_blue = np.where(thresh2*bin_img)

    hist_full_brown= cv2.calcHist([img_grey_mat[index_brown]],[0],None,[256],[0,256])
    hist_full_blue = cv2.calcHist([sat_matrix[index_blue]],[0],None,[256],[0,256])

    # Clip the brown histogram
    hit = hist_full_brown[:,0]
    hit = np.convolve(hit,[1,1,1,1,1,1,1,1,1,1]) # smooth
    max_loc = np.argmax(hit)
    clip_min = 0
    for i in range(max_loc+10,200,1):
        change =hit[i]-hit[i-5]
        print(change)
        if change > 0:
            clip_min = i
            break
    print(clip_min)

    mask_bright_brown = img_grey_mat[:,:]<clip_min
    index_brown_bright_not = np.where(mask_bright_brown*thresh1*bin_img)
    browniest = np.zeros(init_shape)
    browniest[index_brown_bright_not] = img[index_brown_bright_not]

    # Clip the blue histogram
    hit = hist_full_blue[:,0]
    hit = np.convolve(hit,[1,0,-1/2])# smooth
    #hit = np.convolve(hit,[1,1,1,1,1,1,1,1,1])

    clip_min = 0
    for i in range(255,5,-1):
        change =hit[i]-hit[i-5]
        print(change)
        if change > 10:
            clip_min = i
            break
    print(clip_min)

    mask_bright_blue = sat_matrix[:,:]>21
    index_blue_bright_not = np.where(mask_bright_blue*thresh2*bin_img)
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
    show_segmented_result(img, close_img_healthy, close_img_cancer)

    # Display tool for analyzing resulted masks and histograms
    show_masks_and_histograms(close_img_healthy, thresh1*bin_img, [hist_full_blue])
    show_masks_and_histograms(blueiest, close_img_healthy, [hist_full_blue])


# Load the sample image
def load_image(filename):
    img = cv2.imread(filename)
    print "read the image"
    return img, img.shape


def get_grey_and_hsv(img):
    # Get grey for brightness evaluation (matrix and array)
    img_gray_mat = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray_arr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray_arr.shape = (1,img_gray_arr.shape[0]*img_gray_arr.shape[1])
    # Convert to hsv and get channels
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img_gray_mat, img_gray_arr, hsv


def local_minimum(hist):
    hit = hist[:,0]
    max_loc = np.argmax(hit)
    for i in range(max_loc,5,-1):
        change =hit[i]-hit[i-5]
        if change < 0:
            return i


def show_segmented_result(original_img, closed_heathly, closed_cancer):
    # We load with opencv and plot with pyplot (BGR->RGB)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    plt.imshow(original_img)
    plt.contour(closed_heathly, [0.5], linewidths=1, colors='g')
    plt.contour(closed_cancer, [0.5], linewidths=1, colors='r')
    plt.show()


def show_masks_and_histograms(masks1, mask2, histograms):
    plt.subplot(221), plt.imshow(masks1)#, plt.contour(masks1, [0.5], linewidths=1, colors='r')
    plt.subplot(222), plt.imshow(mask2,'gray')
    plt.subplot(212), plt.plot(histograms[0], color = "b")#, plt.plot(hit, color = "k")
    # Brown stuff
    #plt.subplot(212), plt.plot(hist_full_brown, color = "k"),plt.axvline(clip_min), plt.plot(hit, color="pink")#, plt.plot(hist_full_sat, color="green")
    plt.xlim([0,256])
    plt.show()


def save_result(image, filename):
    cv2.imwrite(filename, image)


if __name__ == '__main__':
    main()
