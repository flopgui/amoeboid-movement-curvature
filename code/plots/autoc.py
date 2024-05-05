
import numpy as np
import sys
import matplotlib.pyplot as plt

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
    curvs = np.load(dir+'/menger_curv.npy') - 1/6
    # curvs = np.load(dir+'/nv.npy')
    acurvs = np.load(dir+'/menger_curv.npy') - 1/6
    perimeters = np.load(dir+'/perimeters.npy')
    print(curvs.shape)

    maxangle = np.argmax((acurvs ** 3).sum(axis=0))
    if dir == '../amoeba/maxcurv/6_0' or dir == '../amoeba/maxcurv/7_0':
        maxangle += 180
    curvs = np.roll(curvs, -maxangle+curvs.shape[1]//2, axis=1)
    print(maxangle)

    mean_step = perimeters.mean()/180
    print(mean_step)

    autoc = []
    dts = np.arange(0,200,2)
    # dts = np.arange(0,1000,5)
    # dts = np.arange(0,1000,5)
    # dts = np.arange(0,120,3)
    maxdt = -1
    for idt, dt in enumerate(dts):
        autoc.append(autocorrelation(curvs[800:,:], 0, dt, perimeters))
        # autoc.append(autocorrelation(curvs[800:,:], dt, 0, perimeters))
        print(dt, autoc[-1])
        if maxdt == -1 and autoc[-1] > 0.05:
            continue
        if autoc[-1] >= autoc[maxdt]:
            maxdt = idt
    # np.save(f'{dir}/autoc_curv_time.npy', autoc)
    np.save(f'{dir}/autoc_time.npy', autoc)
    # autoc = np.load(f'{dir}/autoc_curv_time.npy')
    plt.plot(dts/5, autoc, label="r_κ = " + dir[-3:].replace('_','.'))
    # plt.plot(dts/5, autoc)
    # plt.plot(dts*mean_step, autoc)
    plt.plot(dts[maxdt]/5, autoc[maxdt], 'o', c='k')
    # plt.show()

    # autoc = []
    # dxs = np.arange(0,90,3)
    # for dx in dxs:
    #     autoc.append(autocorrelation(curvs[500:,:], dx, 0, perimeters))
    #     print(dx, autoc[-1])
    # plt.plot(dxs, autoc)


    # autoc = []
    # dxs = np.arange(-45,45+1,1)
    # # dxs = np.arange(-45,45+1,3)
    # dts = np.arange(0,200+1,5)
    # # dts = np.arange(0,100+1,10)
    # # dts = np.arange(0,100,50)
    # for i,dx in enumerate(dxs):
    #     autoc.append([])
    #     for j,dt in enumerate(dts):
    #         autoc[i].append(autocorrelation(curvs[1000:,90:180], dx, dt, perimeters, rotate=False))
    #         autoc[i][-1] += autocorrelation(curvs[1000:,0:90], dx, dt, perimeters, rotate=False)
    #         print(dx, dt, autoc[-1][-1])
    # # plt.plot(dxs, autoc)
    # autoc = np.array(autoc)
    # np.save(f'{dir}/autoc_f.npy', autoc)
    # print(autoc.shape, dxs.shape, dts.shape)
    # # dtsd = (dts[1] - dts[0])/2
    # # dxsd = (dxs[1] - dxs[0])/2
    # # plt.imshow(autoc, extent=[dts[0]-dtsd,dts[-1]+dtsd,dxs[0]-dxsd,dxs[-1]-dxsd], vmin=-0.1, vmax=0.1, cmap='RdBu')
    # # plt.show()

plt.legend()
plt.xlabel("Time (s)")
# plt.xlabel("Distance (µm)")
plt.ylabel("Autocorrelation")
# plt.ylim(-0.2,0.5)
plt.savefig("fourier.png", bbox_inches='tight')
plt.show()
