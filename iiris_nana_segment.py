import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage import img_as_ubyte


def main(filename):
    # Get the image and its initial shape
    img, init_shape = load_image(filename) # 70-100

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

    # Collect the non-background pixel values in different channels
    '''indexes = np.where(bin_img)
    img_no_bg = img_grey_arr[indexes]
    hue_no_bg = hue[indexes]
    sat_no_bg = sat[indexes]

    hist_full_gray = cv2.calcHist([np.array(img_no_bg)],[0],None,[256],[0,256])
    hist_full_hue = cv2.calcHist([np.array(hue_no_bg)],[0],None,[256],[0,256])
    hist_full_sat = cv2.calcHist([np.array(sat_no_bg)],[0],None,[256],[0,256])'''

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
    #show_masks_and_histograms(no_bg_mask, cv2.cvtColor(img, cv2.COLOR_BGR2RGB), [hist, hist])
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

    '''print("Max loc 1: "+str(max_loc1))
    print(max_loc2)
    print(clip_min)'''

    mask_bright_brown = img_grey_mat[:,:]<clip_min
    index_brown_bright_not = np.where(mask_bright_brown*thresh_brown*no_bg_mask)
    browniest = np.zeros(init_shape)
    browniest[index_brown_bright_not] = img[index_brown_bright_not]

    '''# Ellipse fitting
    bin4morf = np.uint8(browniest[:,:,0]>0.5)*1
    gray_blur = cv2.GaussianBlur(bin4morf, (15, 15), 0)
    contours, hierarchy = cv2.findContours(gray_blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        #if area < 2000 or area > 4000:
            #continue

        #if len(cnt) < 5:
            #continue

        ellipse = cv2.fitEllipse(cnt)
        cv2.ellipse(browniest, ellipse, (0,255,0), 2)


    #ellipses = cv2.fitEllipse(bin4morf)
    #cv2.ellipse(bin4morf,ellipses,(0,255,0),2) # mby I don't need edge detection here...
    show_masks_and_histograms(closing, gray_blur, [hist_full_brown, hist_brown_modified])
    #show_masks_and_histograms(thresh_brown*no_bg_mask, browniest, [hist_full_brown, hist_brown_modified])'''

    # Clip the blue histogram
    hist_blue_modified = hist_full_blue[:,0]
    hist_blue_modified = np.convolve(hist_blue_modified,[1,1,1])# smooth
    max_loc = np.argmax(hist_blue_modified)
    clip_min = 2*max_loc
    #print(clip_min)

    mask_bright_blue = sat_matrix[:,:]>clip_min
    index_blue_bright_not = np.where(mask_bright_blue*thresh_blue*no_bg_mask)
    blueiest = np.zeros(init_shape)
    blueiest[index_blue_bright_not] = img[index_blue_bright_not]

    #show_masks_and_histograms(thresh_blue*no_bg_mask, blueiest, [hist_full_blue, hist_blue_modified])

    # Opening and Closing
    bin4morf_h = (blueiest>0.5)
    open_img = ndimage.binary_opening(bin4morf_h[:,:,1], structure=np.ones((10,10)))
    # Remove small black hole
    close_img_healthy = ndimage.binary_closing(open_img)
    close_img_healthy = img_as_ubyte(close_img_healthy)
    kernel = np.ones((6,6),np.uint8)
    erode = cv2.erode(close_img_healthy,kernel,iterations = 2)
    #show_masks_and_histograms(close_img_healthy, erode, [hist_full_brown, hist_brown_modified])
    _, contours, hierarchy = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    healthy_cell_estimation = count_ellipses(contours)
    #print("HEALTHY: "+str(healthy_cell_estimation))

    '''blueiest2display = cv2.resize(blueiest, (0,0), fx=0.3, fy=0.3, interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Ellipses", blueiest2display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

    # Opening and Closing
    bin4morf_c = (browniest>0.5)
    open_img = ndimage.binary_opening(bin4morf_c[:,:,1])
    # Remove small black hole
    close_img_cancer = ndimage.binary_closing(open_img)
    close_img_cancer = img_as_ubyte(close_img_cancer)
    kernel = np.ones((6,6),np.uint8)
    erode = cv2.erode(close_img_cancer,kernel,iterations = 2)
    #show_masks_and_histograms(close_img_cancer, erode, [hist_full_brown, hist_brown_modified])
    _, contours, hierarchy = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cancer_cell_estimation = count_ellipses(contours)
    #print("CANCER: "+str(cancer_cell_estimation))

    cancer_mask_total = sum(sum(bin4morf_c[:,:,0]))
    healthy_mask_total = sum(sum(bin4morf_h[:,:,0]))
    mask_estimation = float(cancer_mask_total)/(cancer_mask_total+healthy_mask_total)*100

    ellipse_estimation = float(cancer_cell_estimation)/(cancer_cell_estimation+healthy_cell_estimation)*100

    print(filename.split("/")[-1] + ", " + str(round(ellipse_estimation,2))+ ", " + str(round(mask_estimation,2)))

    hue.shape = (init_shape[0], init_shape[1])
    sat.shape = (init_shape[0], init_shape[1])

    # Display original image with red contour around healthy and green contour around cancer cells
    show_segmented_result(img, close_img_healthy, close_img_cancer)

    # Display tool for analyzing resulted masks and histograms
    #show_masks_and_histograms(close_img_healthy, thresh_brown*no_bg_mask, [hist_full_blue])
    #show_masks_and_histograms(blueiest, close_img_healthy, [hist_full_blue, hist_blue_modified])


# Load the sample image
def load_image(filename):
    img = cv2.imread(filename)
    #print "read the image" + filename
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


def show_segmented_result(original_img, closed_heathly, closed_cancer):
    # We load with opencv and plot with pyplot (BGR->RGB)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    plt.imshow(original_img)
    plt.contour(closed_heathly, [0.5], linewidths=0.7, colors='g')
    plt.contour(closed_cancer, [0.5], linewidths=0.7, colors='r')
    plt.savefig(sys.argv[1]+"_out.png", dpi=700)


def show_masks_and_histograms(masks1, mask2, histograms):
    plt.subplot(221), plt.imshow(masks1,'gray')#, plt.contour(masks1, [0.5], linewidths=1, colors='r')
    plt.subplot(222), plt.imshow(mask2, 'gray')
    plt.subplot(212), plt.plot(histograms[0], color = "b"), plt.plot(histograms[1], color = "r")#, plt.plot(hit, color = "k")
    # Brown stuff
    #plt.subplot(212), plt.plot(hist_full_brown, color = "k"),plt.axvline(clip_min), plt.plot(hit, color="pink")#, plt.plot(hist_full_sat, color="green")
    plt.xlim([0,256])
    plt.show()


def save_result(image, filename):
    cv2.imwrite(filename, image)


def count_ellipses(contours):
    nr_ellipses = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10 or area > 4000:
            continue

        if len(cnt) < 5:
            continue

        ellipse = cv2.fitEllipse(cnt)
        # Handle merged cells
        if area<1000:
            nr_ellipses+=1
        elif area<2000:
            nr_ellipses+=2
        elif area<3000:
            nr_ellipses+=3
        elif area<4000:
            nr_ellipses+=4

        #cv2.ellipse(blueiest, ellipse, (0,255,0), 2)

    return nr_ellipses


# Sobel edge detection function
def sobel(img):
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    img = cv2.GaussianBlur(img,(3,3),0)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Gradient-X
    grad_x = cv2.Sobel(gray,ddepth,1,0,ksize = 3, scale = scale, delta = delta,borderType = cv2.BORDER_DEFAULT)

    # Gradient-Y
    grad_y = cv2.Sobel(gray,ddepth,0,1,ksize = 3, scale = scale, delta = delta, borderType = cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)   # converting back to uint8
    abs_grad_y = cv2.convertScaleAbs(grad_y)

if __name__ == '__main__':
    import sys
    main(sys.argv[1])
