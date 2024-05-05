
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import datetime
import pickle
from PIL import Image
from skimage import measure, segmentation, morphology
import gzip
import scipy.ndimage
from scipy.interpolate import interp1d, RegularGridInterpolator
from numpy import fft
from skimage.io import imread, imshow, imsave
from skimage.color import rgb2gray, label2rgb
import skimage

DX = 0.15
DT = 0.002 * 100
RESOLUTION = 180

def find_smooth_contour(u, resolution):
        contours = measure.find_contours(u, level=0.1)
        if len(contours) > 1:
            pass
            # print("Multiple contours detected!")
        longest = 0
        for i,c in enumerate(contours):
            if len(c) > len(contours[longest]):
                longest = i
        contour = contours[longest][:-1]
        contour_angle = np.arctan2(contour[:, 1] - u.shape[1]/2, contour[:, 0] - u.shape[0]/2)
        angle_shift = np.argmax(np.diff(contour_angle) > 0)
        contour = np.roll(contour, -angle_shift-1, axis=0)

        contour_dist = np.sqrt(np.diff(contour[:, 1], append=contour[0,1]) ** 2 + np.diff(contour[:, 0], append=contour[0,0]) ** 2)
        perimeter = np.sum(contour_dist)
        xinterp = interp1d(np.concatenate([np.zeros(1), np.cumsum(contour_dist)/perimeter]), np.concatenate([contour[:, 0], contour[0:1,0]]))
        yinterp = interp1d(np.concatenate([np.zeros(1), np.cumsum(contour_dist)/perimeter]), np.concatenate([contour[:, 1], contour[0:1,1]]))
        s = np.linspace(0,1,resolution,endpoint=False)
        smooth_contour = np.array([xinterp(s), yinterp(s)]).T
        return smooth_contour

def menger_curvature(p, q, r):
    tr_area = 0.5 * np.cross([p[0]-q[0],p[1]-q[1],0], [q[0]-r[0],q[1]-r[1],0])[2]
    return -4 * tr_area / (np.linalg.norm(p-q) * np.linalg.norm(q-r) * np.linalg.norm(r-p))

def normal_vector(p, q, r):
    n = np.array([-r[1] + p[1], r[0] - p[0]])
    return n / np.linalg.norm(n)

def centroid(u):
    xg, yg = np.meshgrid(np.arange(0.5, 0.5 + u.shape[1]), np.arange(0.5, 0.5 + u.shape[0]))
    xc = np.sum(u * yg) / u.sum()
    yc = np.sum(u * xg) / u.sum()
    return np.array([xc, yc])


def autocorrelation(u, ds, dt, perimeter):
    um = np.mean(u)
    num = 0
    den = 0
    for t in range(u.shape[0]):
        for s in range(u.shape[1]):
            if t + dt >= 0 and t + dt < u.shape[0]:
                num += (u[t+dt, (s+ds) % u.shape[1]] - um) * (u[t, s] - um) * perimeter[t]  # * DT * (1/RESOLUTION)
            den += (u[t, s] - um) ** 2 * perimeter[t]  # * DT * (1/RESOLUTION)
    if den == 0:
        return 0
    return num / den

def load_file_model(dir):
    phi_file = gzip.open(os.path.join(dir,"phi.txt.gz"), 'r')
    for ts, row in enumerate(phi_file):
        phi = np.array([[float(i) for i in l.split()] for l in row.decode().split(';')])
        yield phi

def load_file_real(dir):
    filelist = os.popen(f'ls {dir} | grep png').read().split('\n')[:-1]
    # filelist = os.popen(f'ls {dir} | grep jpg').read().split('\n')[:-1]
    for filei, filepath in enumerate(filelist):
        phi = np.array(imread(os.path.join(dir,filepath))/255)
        if len(phi.shape) > 2:
            phi = phi[:,:,0]
        yield phi


def segment_film(dir, resolution):
    # contour = np.empty((len(files), 2 * resolution + 1))

    try:
        shifts_file = open(os.path.join(dir,'shift.txt'))
    except:
        shifts_file = None
    phis = load_file_model(dir)
    menger_curvs = []
    nvs = []
    perimeters = []
    contour = None
    sx, sy = 0, 0
    cent = np.array([100, 100])
    contours = []
    centroids = []
    allcontours = []
    for ts, phi in enumerate(phis):
        if shifts_file:
            sx, sy = (float(x) for x in shifts_file.readline().split())
        else:
            sx, sy = 0, 0
        if ts < 190/DT: continue
        if ts > 215/DT: break
        if ts % int(5/DT) != 0: continue
        if ts % 4 == 0: print(ts)
        prev_contour = contour
        contour = find_smooth_contour(phi, resolution)
        contour_dist = np.sqrt(np.diff(contour[:, 1], append=contour[0,1]) ** 2 + np.diff(contour[:, 0], append=contour[0,0]) ** 2)
        perimeters.append(np.sum(contour_dist))
        ps = (sx, sy)
        abs_contour = np.copy(contour)
        contour[:, 0] += sx
        contour[:, 1] += sy
        prev_cent = cent
        cent = centroid(phi) + np.array([sx, sy])
        vcent = cent - prev_cent
        allcontours.append(contour)
        contours.append(contour)
        centroids.append(cent)
        # curv_interp = RegularGridInterpolator((np.arange(0.5, phi.shape[0]+0.5), np.arange(0.5, phi.shape[1]+0.5)), curv)
        # curv_c = curv_interp(abs_contour)
        # curvs.append(curv_c)
        menger_curv = []
        nv = []
        for i in range(len(contour)):
            menger_curv.append(menger_curvature(contour[i-1], contour[i], contour[(i+1)%len(contour)]) / DX)
        menger_curvs.append(menger_curv)
        if prev_contour is not None:
            for i in range(len(contour)):
                normal = normal_vector(contour[i-1], contour[i], contour[(i+1)%len(contour)])
                nvel = np.dot(contour[i] - prev_contour[i] - vcent, normal) * DX / DT
                nv.append(nvel)
            if (sx, sy) == ps or True:
                nvs.append(nv)
    menger_curvs = np.array(menger_curvs)
    nvs = np.array(nvs)
    return contours, menger_curvs, centroids, allcontours

contours, curvs, centroids, allcontours = segment_film(sys.argv[1], 180)
# cmap = plt.get_cmap('jet')
cmap = plt.get_cmap('gist_rainbow')
print(len(contours))
for ic, contour in enumerate(allcontours):
    # for i in range(contour.shape[0]-1):
    #     plt.plot(DX*contour[:,0], DX*contour[:,1], c='k')
# for ic, contour in enumerate(contours):
    for i in range(contour.shape[0]-1):
        plt.plot(-DX*(contour[i:i+2,0]-centroids[ic][0]), DX*(contour[i:i+2,1]-centroids[ic][1]), c=cmap(ic/(len(contours)-0.5)))
        # plt.plot(DX*(contour[i:i+2,0]-centroids[0][0]), DX*(contour[i:i+2,1]-centroids[0][1]), c=cmap(ic/len(contours)))
        # if ic % 2 == 0:
        #     plt.plot(DX*contour[i:i+2,0], DX*contour[i:i+2,1], c=cmap(curvs[ic,i]+0.5))
        # else:
        #     plt.plot(DX*contour[i:i+2,0], DX*contour[i:i+2,1], c='#bbbbbb')
plt.gca().set_aspect('equal')
plt.xlabel('Distance (µm)')
plt.ylabel('Distance (µm)')
# plt.axis('off')
plt.savefig('outlines.png')
# plt.show()
