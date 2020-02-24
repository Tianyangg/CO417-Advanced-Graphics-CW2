import sys
import numpy as np
import math
import copy

from PNM import *

### dealing with HDR image
# exponential scale on HDR
def hdr_load_stop(f, stop):
    multiplier = np.power(2, stop)
    f = np.multiply(f, multiplier)
    f[f > 1] = 1 # deal with numbers larger than 255
    return f


def tone_gamma(F, gamma):
    return np.multiply(np.power(F, (1./gamma)), 255)

def Gamma_Exp_Scale(f, gamma, stop):
    tmp = copy.deepcopy(f)
    tmp = tmp*(2.**stop)
    tmp[tmp > 1.] = 1.
    tmp = tmp**(1./gamma)
    tmp = np.uint8(np.round(tmp*255))
    return tmp
    # writePPM('../part1-outputs/HDR_gamma_{}_stop_{}.ppm'.format(gamma, stop), tmp)

def MC_sample(sample_num, show_sample_only):
    img = loadPFM('../GraceCathedral/grace_latlong.pfm')
    stop = 6
    gamma = 2.2
    # deal with overflow numbers
    img[img > 255.] = 255.
    img[img < 0.] = 0.
    # img = img /255.
    img = img / (np.max(img))  # normalize the image to scale 0 - 1

    F = copy.deepcopy(img)  # make a copy without reference
    intensity = np.sum(F, 2, dtype=float) / 3  # I = (RGB/3)

    # scale intensity by solid angle
    height, width, c = img.shape
    scale = np.array([np.sin(float(h)/(height - 1.0)* np.pi) for h in range(height)])
    intensity = np.multiply(scale, intensity.T).T  # element wise dot product with the scale, do for every colum

    if show_sample_only:
        img_demo = np.zeros((height, width, c))
    for i in range(sample_num):

        row_i = sample_row(intensity)
        col_j = sample_col(intensity, row_i)

        if show_sample_only:
            img_demo[max(0, row_i - 2): min(height, row_i + 3), max(0, col_j - 2): min(col_j + 3, width)] = img[row_i, col_j]

        else:
            # set pixel around to (0, 0, 1), and deal with potential out of range error
            img[max(0, row_i - 2): min(height, row_i + 3), max(0, col_j - 2), :] = [0, 0, 1]
            img[max(0, row_i - 2): min(height, row_i + 3), min(col_j + 2, width - 1), :] = [0, 0, 1]
            img[max(0, row_i - 2), max(0, col_j - 2): min(col_j + 3, width), :] = [0, 0, 1]
            img[min(height - 1, row_i + 2), max(0, col_j - 2): min(col_j + 3, width), :] = [0, 0, 1]

    if show_sample_only:
        img_demo = Gamma_Exp_Scale(img_demo, gamma, stop)
        writePPM('../SAMPLE_Only_MCw_{}_s{}_g{}.ppm'.format(sample_num, stop, gamma), img_demo)
    else:
        img = Gamma_Exp_Scale(img, gamma, stop)
        writePPM('../MCw_{}_s{}_g{}.ppm'.format(sample_num, stop, gamma), img)




def sample_row(image):
    column_sum = np.sum(image, 1)  # return a 1d array sum
    cdf = np.cumsum(column_sum)/np.sum(column_sum)  # GET A CDF
    rand = np.random.uniform(0, 1, 1)
    return np.argwhere(cdf >= rand)[0][0]  ## return the row index


def sample_col(image, i):
    row_selected = image[i, :]
    cdf = np.cumsum(row_selected)/np.sum(row_selected)
    rand = np.random.uniform(0, 1, 1)
    return np.argwhere(cdf >= rand)[0][0]



def MC_sample_new(sample_num, img, inten):
    # img = loadPFM('../GraceCathedral/grace_latlong.pfm')
    # # print(img[img>200])
    # img[img > 255] = 255
    # img = img/255
    #
    #
    # F = copy.deepcopy(img)  # copy without reference
    # F[F > 255] = 255
    # intensity = np.sum(F, 2)/3 # I = (RGB/3)
    #
    # # normalize intensity
    # intensity = intensity/255


    # scale intensity by solid angle
    height, width, c = img.shape
    # scale = np.array([np.sin(h/(height - 1.0)) * np.pi for h in range(height)])
    # intensity = np.multiply(scale, intensity.T).T
    # # img = img/255
    for i in range(sample_num):
        # sample given tow
        row_i = sample_row(inten)
        col_j = sample_col(inten, row_i)
        # print(row_i, col_j)


        # set pixel around to (0, 0, 1)

        img[max(0, row_i - 2): min(height, row_i + 3), max(0, col_j - 2), :] = [0, 0, 1]
        img[max(0, row_i - 2): min(height, row_i + 3), min(col_j + 2, width - 1), :] = [0, 0, 1]
        img[max(0, row_i - 2), max(0, col_j - 2): min(col_j + 3, width), :] = [0, 0, 1]
        img[min(height - 1, row_i + 2), max(0, col_j - 2): min(col_j + 3, width), :] = [0, 0, 1]


    # img = hdr_load_stop(img, 6)
    # img = tone_gamma(img, 2.2)
    # writePPM('../MCw_{}.ppm'.format(sample_num), img.astype(np.uint8))


def median_cut(max_cut):

    img = loadPFM('../GraceCathedral/grace_latlong.pfm')
    # print(img[img>200])
    img[img > 255] = 255
    img = img / 255
    F = copy.deepcopy(img)  # copy without reference
    F[F > 255] = 255

    intensity = np.average(F, weights=[0.2125, 0.7154, 0.0721], axis=2)  # Get intensity I = (RGB/3)


    # normalize intensity
    intensity = intensity
    print(intensity[50:100, 50])

    height, width, c = img.shape

    scale = np.array([np.sin((float(h) / float(height-1)) * np.pi) for h in range(height)])
    print("!!!!!", scale)
    intensity = np.multiply(scale, intensity.T).T

    cut_recur(intensity, img, max_cut)
    img = hdr_load_stop(img, 6)
    img = tone_gamma(img, 2.2)
    writePPM('../MCw_.ppm', img.astype(np.uint8))

def cut_recur(intens, image, depth):
    if depth == 0:
        hh, ww = intens.shape
        MC_sample_new(10, image, intens)

        # image[max(0, int(hh/2) - 2): min(hh, int(hh/2) + 3), max(0, int(ww/2) - 2),:] = [0, 0, 1]
        # image[max(0, int(hh/2) - 2): min(hh, int(hh/2) + 3), min(ww - 1, int(ww / 2) + 2), :] = [0, 0, 1]
        # image[max(0, int(hh/2) - 2), max(0, int(ww/2) - 2): min(ww, int(ww / 2) + 3) , :] = [0, 0, 1]
        # image[min(hh - 1, int(hh/2) + 2), max(0, int(ww/2) - 2): min(ww, int(ww / 2) + 3), :] = [0, 0, 1]

        return 0

    cut, d = sample_median(intens)
    if d == 0:
        image[cut, :, :] = [0, 1, 0]
        cut_recur(intens[0:cut,:], image[0:cut, :, :], depth - 1)
        cut_recur(intens[cut::,:], image[cut::, :, :], depth - 1)
        print(np.sum(intens[0:cut, :]), np.sum(intens[cut::, :]))
    else:
        image[:, cut, :] = [0, 1, 0]
        cut_recur(intens[:, 0:cut], image[:, 0:cut, :], depth - 1)
        cut_recur(intens[:, cut::], image[:, cut::, :], depth - 1)
        # print(intens[0:10, 0:10])
        print(np.sum(intens[:, 0:cut]), np.sum(intens[:, cut::]))


def sample_median(imag):
    # get longer dimension
    h, w = imag.shape
    if h > w:
        dim_cut = 1
        dim = 0
    else:
        dim_cut = 0
        dim = 1

    # sum over the dim
    intensity_1d = np.sum(imag, dim_cut)
    temp = np.cumsum(intensity_1d)
    cum_intensity = temp # /np.sum(intensity_1d)
    # print(cum_intensity)

    return np.argwhere(cum_intensity > 0.5*cum_intensity[-1])[0][0], dim

# MC samples
if '__main__' == __name__:

    # MC_sample(64, False)
    # MC_sample(256, False)
    # MC_sample(1024, False)
    MC_sample(256, True)
    # median_cut(6)
