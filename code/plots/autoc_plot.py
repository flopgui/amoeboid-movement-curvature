

import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.interpolate

import matplotlib
matplotlib.rcParams.update({'font.size': 15})

def autocorrelation(u, ds, dt, perimeter, rotate=True):
    um = np.mean(u)
    num = 0
    den = 0
    for t in range(u.shape[0]):
        for s in range(u.shape[1]):
            if t + dt >= 0 and t + dt < u.shape[0]:
                if rotate:
                    num += (u[t+dt, (s+ds) % u.shape[1]] - um) * (u[t, s] - um) * perimeter[t]  # * DT * (1/RESOLUTION)
                else:
                    if s + ds >= 0 and s + ds < u.shape[1]:
                        num += (u[t+dt, s+ds] - um) * (u[t, s] - um) * perimeter[t]  # * DT * (1/RESOLUTION)
            den += (u[t, s] - um) ** 2 * perimeter[t]  # * DT * (1/RESOLUTION)
    if den == 0:
        return 0
    return num / den

dirs = sys.argv[1:]

for dir in dirs:
    print(dir)
    # curvs = np.load(dir+'/menger_curv.npy') - 1/6
    curvs = np.load(dir+'/nv.npy')
    acurvs = np.load(dir+'/menger_curv.npy') - 1/6
    perimeters = np.load(dir+'/perimeters.npy')
    # print(curvs.shape)

    maxangle = np.argmax((acurvs ** 3).sum(axis=0))
    if dir == '../amoeba/maxcurv/6_0' or dir == '../amoeba/maxcurv/7_0':
        maxangle += 180
    curvs = np.roll(curvs, -maxangle+curvs.shape[1]//2, axis=1)
    # print(maxangle)

    mean_step = perimeters[0:200].mean()/360

    # autoc = []
    # dts = np.arange(0,200,5)
    # for dt in dts:
    #     autoc.append(autocorrelation(curvs[500:,:], 0, dt, perimeters))
    #     print(dt, autoc[-1])
    # plt.plot(dts, autoc)

    # autoc = []
    # dxs = np.arange(0,90,3)
    # for dx in dxs:
    #     autoc.append(autocorrelation(curvs[500:,:], dx, 0, perimeters))
    #     print(dx, autoc[-1])
    # plt.plot(dxs, autoc)


    autoc = []
    dxs = np.arange(-45,45+1,1)
    # dxs = np.arange(-45,45+1,3)
    dts = np.arange(0,200+1,5)
    # dts = np.arange(0,100+1,10)
    # dts = np.arange(0,100,50)
    autoc = np.load(f'{dir}/autoc_f.npy')
    # plt.plot(dxs, autoc)
    autoc = np.array(autoc)
    autoc_sym = np.concatenate((np.flip(autoc[::-1], axis=1), autoc), axis=1)
    fig, ax = plt.subplots(2,2,gridspec_kw={'width_ratios': [0.94,0.03]})
    plot1 = ax[0][0].imshow(curvs[1000:].T, extent=[0,300,0,360], vmin=-0.1, vmax=0.1, cmap='PiYG')
    ax[0][0].set_aspect(1/3)
    plt.colorbar(plot1, cax=ax[0][1])
    dxs = dxs / (curvs.shape[1]/500)
    dts = dts / (curvs.shape[0]/360)
    dtsd = (dts[1] - dts[0])/2
    dxsd = (dxs[1] - dxs[0])/2
    plot2 = ax[1][0].imshow(autoc_sym, extent=[-dts[-1]-dtsd,dts[-1]+dtsd,dxs[0]-dxsd,dxs[-1]-dxsd], vmin=-0.2, vmax=0.2, cmap='PiYG')
    plot2 = ax[1][0].imshow(autoc_sym, extent=[-dts[-1]-dtsd,dts[-1]+dtsd,dxs[0]-dxsd,dxs[-1]-dxsd], vmin=-0.02, vmax=0.02, cmap='PiYG')
    ax[1][0].set_aspect(1/10)
    plt.colorbar(plot2, cax=ax[1][1])

    # print(autoc.shape)
    interp = scipy.interpolate.RegularGridInterpolator((dxs, dts), autoc)
    # print(interp((0,1)))

    # angles = np.concatenate((np.arange(-90,-45,1), np.arange(45,90,1)))
    # angles = angles / 180 * np.pi
    # angles = np.arctan(np.concatenate((np.arange(-20,-1,1), [0], np.arange(1,20,1))))
    minslope = 8.0
    maxslope = 8.0001
    stepslope = 0.5
    angles = np.arctan(np.concatenate((np.arange(-maxslope,-minslope,stepslope), np.arange(minslope,maxslope,stepslope))))
    # angles = np.arctan(np.arange(-maxslope,-minslope,stepslope))
    sums = []
    for iangle, angle in enumerate(angles):
        if (iangle+1)%100 == 0:
            pass
            # print(f"{iangle/len(angles)*100:.1f}%")
        slope = np.tan(angle)
        minv = min(dts[-1], dxs[-1]/np.abs(slope))
        x = np.linspace(0, minv, 1000)[1:-1]
        ax[1][0].plot(x,x*slope)
        # vx = np.array((x, slope*x)).T
        # print(vx)
        # print(vx.shape)
        # vals = interp(vx[1:8])

        vals = np.array([interp((slope*xi, xi)) for xi in x])
        sums.append((vals).sum()/len(vals))
    print(sums)
    sums = np.array(sums)
    # ax[2].plot(np.tan(angles), sums)

    maxsum = np.argmax(sums[:len(sums)//2])
    slope1 = slope = np.tan(angles[maxsum])
    # print(slope)
    minv = min(dts[-1], dxs[-1]/np.abs(slope))
    x = np.linspace(0, minv, 1000)[1:-1]
    ax[1][0].plot(x, slope * x, c='k')
    ax[1][0].plot(-x, -slope * x, c='k')
    maxsum = np.argmax(sums[len(sums)//2:]) + len(sums)//2
    slope2 = slope = np.tan(angles[maxsum])
    # print(slope)
    minv = min(dts[-1], dxs[-1]/np.abs(slope))
    x = np.linspace(0, minv, 1000)[1:-1]
    ax[1][0].plot(x, slope * x, c='k')
    ax[1][0].plot(-x, -slope * x, c='k')
    print((abs(slope1) + abs(slope2))/2)
    print((abs(slope1) + abs(slope2))*mean_step)

    # ax[0][0].set_xlabel('Time (s)')
    ax[0][0].set_ylabel('Angle (deg)')
    ax[1][0].set_xlabel('Time (s)')
    ax[1][0].set_ylabel('Angle (deg)')
    ax[0][1].set_ylabel('Relative normal vel. (µm/s)')
    ax[1][1].set_ylabel('Autocorrelation')

    print(autoc_sym.shape)
    mp = autoc_sym.shape[1] // 2
    # ax[2].plot(np.concatenate((np.flip(-dts), dts)), autoc_sym[mp,:])

    # curvs = np.load(dir+'/menger_curv.npy') - 1/6
    # # curvs = np.load(dir+'/nv.npy')
    # acurvs = np.load(dir+'/menger_curv.npy') - 1/6
    # perimeters = np.load(dir+'/perimeters.npy')
    # print(curvs.shape)

    # maxangle = np.argmax((acurvs ** 3).sum(axis=0))
    # if dir == '../amoeba/maxcurv/6_0' or dir == '../amoeba/maxcurv/7_0':
    #     maxangle += 180
    # curvs = np.roll(curvs, -maxangle+curvs.shape[1]//2, axis=1)
    # print(maxangle)

    # autoc = []
    # dts = np.arange(0,200,5)
    # for dt in dts:
    #     autoc.append(autocorrelation(curvs[500:,:], 0, dt, perimeters))
    #     print(dt, autoc[-1])
    # ax[2].plot(dts/5, autoc)

    plt.savefig('fourier_plot.png', bbox_inches='tight')
    # plt.show()

# plt.show()
