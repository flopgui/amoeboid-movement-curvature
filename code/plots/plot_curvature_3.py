

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
# RESOLUTION = 60

# DX = 5/36 # Videos

def find_smooth_contour(u, resolution):
        # contours = measure.find_contours(u, level=0.5)
        contours = measure.find_contours(u, level=0.1)
        if len(contours) > 1:
            pass
            # print("Multiple contours detected!")
        longest = 0
        for i,c in enumerate(contours):
            if len(c) > len(contours[longest]):
                longest = i
        contour = contours[longest][:-1]
        # if ts == 0:
        #     angle = np.arange(0, 2*np.pi, 2*np.pi/180)
        #     init = np.array([len(phi)/2 + 0.3*len(phi)*np.cos(angle), len(phi)/2 + 0.3*len(phi)*np.sin(angle)]).T
        #     contour = segmentation.active_contour(scipy.ndimage.gaussian_filter(phi, 2), init, alpha=0.1, beta=0.1, gamma=0.001)
        #     plt.imshow(phi, cmap=plt.cm.gray)
        #     plt.plot(init[:, 1], init[:, 0], '--r', lw=3)
        #     plt.plot(contour[:, 1], contour[:, 0], '-b', lw=3)
        # else:
        #     angle = np.arctan2(contour[:, 1] - len(phi)/2, contour[:, 0] - len(phi)/2)
        #     init = np.array([contour[:,0] + 0.05*len(phi)*np.cos(angle), contour[:, 1] + 0.05*len(phi)*np.sin(angle)]).T
        #     contour = segmentation.active_contour(segmentation.find_boundaries(phi > 0.5), init, alpha=0.01, beta=0.1, gamma=0.001)
        #     if ts % 1 == 0:
        #         plt.imshow(scipy.ndimage.gaussian_filter(phi, 0.01), cmap=plt.cm.gray)
        #         plt.plot(init[:, 1], init[:, 0], '--r', lw=3)
        #         plt.plot(contour[:, 1], contour[:, 0], '-b', lw=3)

        # Start always at same angle
        contour_angle = np.arctan2(contour[:, 1] - u.shape[1]/2, contour[:, 0] - u.shape[0]/2)
        angle_shift = np.argmax(np.diff(contour_angle) > 0)
        contour = np.roll(contour, -angle_shift-1, axis=0)

        contour_dist = np.sqrt(np.diff(contour[:, 1], append=contour[0,1]) ** 2 + np.diff(contour[:, 0], append=contour[0,0]) ** 2)
        perimeter = np.sum(contour_dist)
        xinterp = interp1d(np.concatenate([np.zeros(1), np.cumsum(contour_dist)/perimeter]), np.concatenate([contour[:, 0], contour[0:1,0]]))
        yinterp = interp1d(np.concatenate([np.zeros(1), np.cumsum(contour_dist)/perimeter]), np.concatenate([contour[:, 1], contour[0:1,1]]))
        s = np.linspace(0,1,resolution,endpoint=False)
        smooth_contour = np.array([xinterp(s), yinterp(s)]).T

        # print(smooth_contour)
        # plt.imshow(u, cmap=plt.cm.gray)
        # for i in range(0, len(contours)):
        #     plt.plot(contours[i][:,1], contours[i][:,0], c='r')
        # plt.plot(smooth_contour[:,1], smooth_contour[:,0])
        # # plt.draw()
        # # plt.waitforbuttonpress(0) # this will wait for indefinite time
        # plt.show()

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
    # curv_file = gzip.open(os.path.join(dir,"curv.txt.gz"), 'r')
    # curvs = []
    menger_curvs = []
    nvs = []
    perimeters = []
    contour = None
    sx, sy = 0, 0
    cent = np.array([100, 100])
    for ts, phi in enumerate(phis):
        if ts % 20 == 0:
            print(ts)
        # if ts == 300:
        #     break
        # row = curv_file.readline()
        # curv = np.array([[float(i) for i in l.split()] for l in row.decode().split(';')])
        prev_contour = contour
        contour = find_smooth_contour(phi, resolution)
        # plt.imshow(phi, cmap=plt.cm.gray)
        # plt.plot(contour[:,1], contour[:,0])
        # plt.plot(contour[:,1], contour[:,0])
        # plt.draw()
        # plt.waitforbuttonpress(0) # this will wait for indefinite time
        # plt.show()
        contour_dist = np.sqrt(np.diff(contour[:, 1], append=contour[0,1]) ** 2 + np.diff(contour[:, 0], append=contour[0,0]) ** 2)
        perimeters.append(np.sum(contour_dist))
        ps = (sx, sy)
        if shifts_file:
            sx, sy = (float(x) for x in shifts_file.readline().split())
        else:
            sx, sy = 0, 0
        abs_contour = np.copy(contour)
        contour[:, 0] += sx
        contour[:, 1] += sy
        prev_cent = cent
        cent = centroid(phi) + np.array([sx, sy])
        vcent = cent - prev_cent
        # plt.imshow(phi, cmap=plt.cm.gray)
        # plt.plot(contour[:,1], contour[:,0])
        # plt.plot(contour[:,1], contour[:,0])
        # plt.plot([phi.shape[0]//2], [phi.shape[1]/2], 'bo')
        # plt.plot([cent[0]], [cent[1]], 'ro')
        # plt.show()
        # continue
        xg, yg = np.meshgrid(np.arange(0, phi.shape[0]+1), np.arange(0, phi.shape[0]+1))
        # curv_interp = RegularGridInterpolator((np.arange(0.5, phi.shape[0]+0.5), np.arange(0.5, phi.shape[1]+0.5)), curv)
        # curv_c = curv_interp(abs_contour)
        # curvs.append(curv_c)
        menger_curv = []
        nv = []
        for i in range(len(contour)):
            menger_curv.append(menger_curvature(contour[i-1], contour[i], contour[(i+1)%len(contour)]) / DX)
        menger_curvs.append(menger_curv)
        if ts > 0:
            for i in range(len(contour)):
                normal = normal_vector(contour[i-1], contour[i], contour[(i+1)%len(contour)])
                nvel = np.dot(contour[i] - prev_contour[i] - vcent, normal) * DX / DT
                nv.append(nvel)
            if (sx, sy) == ps or True:
                nvs.append(nv)
    # curvs = np.array(curvs)
    menger_curvs = np.array(menger_curvs)
    nvs = np.array(nvs)
    print(nvs.shape)

    np.save(dir+'/menger_curv.npy', menger_curvs)
    np.save(dir+'/nv.npy', nvs)
    np.save(dir+'/perimeters.npy', perimeters)

    # menger_curvs = np.load(dir+'/menger_curv.npy')
    # nvs = np.load(dir+'/nv.npy')
    # perimeters = np.load(dir+'/perimeters.npy')
    cut = (0,len(nvs))
    # cut = (220,340)
    # cut = (350,650)
    # cut = (200,700)
    # cut = (80,150)
    menger_curvs = menger_curvs[cut[0]:cut[1]]
    nvs = nvs[cut[0]:cut[1]]
    perimeters = perimeters[cut[0]:cut[1]]

    acs = []
    acs_curv = []
    print(nvs.shape)
    # dts = np.arange(0, 40, 10)
    dts = np.arange(0, 120, 40)
    dss = np.arange(-60, 63, 20)
    # dts = np.arange(0, 120, 30)
    # dss = np.arange(-60, 63, 20)
    # dts = np.arange(0, 9, 3)
    # dss = np.arange(-6, 9, 3)
    for dt in dts:
        acs.append([])
        acs_curv.append([])
        for ds in dss:
            acs[-1].append(autocorrelation(nvs, ds, dt, perimeters))
            acs_curv[-1].append(autocorrelation(menger_curvs, ds, dt, perimeters))
            print(dt, ds, acs[-1][-1], acs_curv[-1][-1])
    acs = np.array(acs)
    acs_curv = np.array(acs_curv)
    np.save(dir+'/ac.npy', acs)
    np.save(dir+'/ac_curv.npy', acs_curv)
    np.save(dir+'/ac_dt.npy', dts)
    np.save(dir+'/ac_ds.npy', dss)

    # acs = np.load(dir+'/ac.npy')
    # acs_curv = np.load(dir+'/ac_curv.npy')
    # dts = np.load(dir+'/ac_dt.npy')
    # dss = np.load(dir+'/ac_ds.npy')

    dts = np.concatenate((-dts[:0:-1], dts))
    acs = np.concatenate((np.flip(acs[:0:-1], axis=1), acs), axis=0)
    acs_curv = np.concatenate((np.flip(acs_curv[:0:-1], axis=1), acs_curv), axis=0)

    # fig,(ax1,ax2) = plt.subplots(2,1)
    # ax1.imshow(nvs.T, cmap='RdBu', vmin=-0.5, vmax=0.5)
    # ax2.imshow(acs.T, cmap='seismic', vmin=-0.8, vmax=0.8)
    # ax2.set_xticks(np.linspace(0, acs.shape[0], len(dts[::6])))
    # ax2.set_xticklabels(dts[::6])
    # ax2.set_yticks(np.linspace(0, acs.shape[1], len(dss[::4])))
    # ax2.set_yticklabels(dss[::4])
    # plt.show()

    fig,(ax1,ax2) = plt.subplots(2,1)
    ax1.imshow(menger_curvs.T, cmap='RdBu', vmin=-1.0, vmax=1.0, extent=[0,len(nvs)/5,0,360])
    ax1.set_aspect(1/3)
    ax1.set_title('Curvature')
    ax1.set_xlabel('t (s)')
    ax1.set_ylabel('angle')
    ax2.imshow(nvs.T, cmap='RdBu', vmin=-1.0, vmax=1.0, extent=[0,len(nvs)/5,0,360])
    ax2.set_aspect(1/3)
    ax2.set_title('Normal velocity')
    ax2.set_xlabel('t (s)')
    ax2.set_ylabel('angle')
    # plt.show()

    # fig,(ax1,ax2) = plt.subplots(2,1)
    # ax1.imshow(nvs.T, cmap='RdBu', vmin=-1.0, vmax=1.0, extent=[0,len(nvs)/5,0,360])
    # ax1.set_aspect(1/3)
    # ax2.imshow(acs.T, cmap='seismic', vmin=-0.4, vmax=0.4)
    # ax2.set_xticks(np.linspace(0, acs.shape[0], len(dts[::6])))
    # ax2.set_xticklabels(dts[::6]/5)
    # ax2.set_yticks(np.linspace(0, acs.shape[1], len(dss[::4])))
    # ax2.set_yticklabels(dss[::4])
    # ax1.set_title('Normal velocity')
    # ax1.set_xlabel('t (s)')
    # ax1.set_ylabel('angle')
    # ax2.set_title('Auto correlation')
    # ax2.set_xlabel('t (s)')
    # ax2.set_ylabel('angle')
    # plt.show()

    # fig,(ax1,ax2) = plt.subplots(2,1)
    # ax1.imshow(menger_curvs.T, cmap='RdBu', vmin=-1.0, vmax=1.0, extent=[0,len(menger_curvs)/5,0,360])
    # ax1.set_aspect(1/3)
    # ax2.imshow(acs_curv.T, cmap='seismic', vmin=-0.8, vmax=0.8)
    # ax2.set_xticks(np.linspace(0, acs_curv.shape[0], len(dts[::6])))
    # ax2.set_xticklabels(dts[::6]/5)
    # ax2.set_yticks(np.linspace(0, acs_curv.shape[1], len(dss[::4])))
    # ax2.set_yticklabels(dss[::4])
    # ax1.set_title('Curvature')
    # ax1.set_xlabel('t (s)')
    # ax1.set_ylabel('angle')
    # ax2.set_title('Auto correlation')
    # ax2.set_xlabel('t (s)')
    # ax2.set_ylabel('angle')
    # plt.show()

    # ft = fft.fft2(acs.T)
    # ft_c = fft.fft2(acs_curv.T)
    # fig,(ax1,ax2) = plt.subplots(2,1)
    # ax1.imshow(np.abs(ft))
    # ax2.imshow(np.abs(ft_c))
    # plt.show()


if sys.argv[1] == '-m':
    for dir in sys.argv[2:]:
        print(dir)
        segment_film(dir, RESOLUTION)
else:
    # timesteps = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    # filelist = os.popen(f'ls {} | grep phi').read().split('\n')[:-1][:timesteps]
    dir = sys.argv[1]
    segment_film(dir, RESOLUTION)
    # with open(dir+'/contour.pickle', 'wb') as f:
    #     pickle.dump(contour, f)

