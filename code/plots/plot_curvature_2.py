

from os import times
import numpy as np
import matplotlib.pyplot as plt
import sys

DX = 0.15

fig = plt.figure()

def read_file(filepath):
    with open(filepath) as file:
        curvs = []
        line = True
        while line:
            line = file.readline()
            curvs.append([float(x) for x in line.split()])
        curvs = curvs[:-1]
    curvs = np.array(curvs)
    return curvs

for i,dir in enumerate(sys.argv[1:]):
    ncols = 1 if len(sys.argv) <= 4 else 2
    ax = fig.add_subplot(len(sys.argv)//ncols,ncols,i+1)
    curvs = read_file(dir+f'/curv_histogram.txt')
    bct = read_file(dir+f'/bct_histogram.txt')
    vn = read_file(dir+f'/vn_histogram.txt')
    vnb = read_file(dir+f'/vnb_histogram.txt')

# shift = read_file(dir+f'/shift.txt')
# centroid = read_file(dir+f'/centroid.txt')
# time = read_file(dir+f'/time.txt')
# act_centroid = (shift + centroid) * DX
# v_centroid = np.diff(act_centroid, axis=0) / np.diff(time, axis = 0)
# v_centroid_proj = [np.zeros(len(vn[0]))]
# for ts in v_centroid:
#     v_centroid_proj.append([np.linalg.norm(ts) * np.cos(np.angle(ts[0] + ts[1] * 1j) - angle) for angle in np.arange(0,2*np.pi,2*np.pi/360)])
# v_centroid_proj = np.array(v_centroid_proj)
# print(v_centroid_proj)

# maxv = np.max(np.abs(curvs))/2
# maxv = 1.0
# fig, ax = plt.subplots(3,1)
# ax[0].imshow(curvs.T, cmap='RdBu', vmin=-maxv, vmax=maxv, extent=[0,len(curvs)/5,0,360])
# ax[0].set_aspect(1/3)
# ax[1].imshow(bct.T, vmin=0, vmax=1, extent=[0,len(bct)/5,0,360])
# ax[0].set_aspect(1/3)
# print(vn.min())
# print(vn.max())
# print(np.abs(vn).mean())
# ax[2].imshow(vn.T, cmap='RdBu', vmin=-0.5, vmax=0.5, extent=[0,len(vn)/5,0,360])
# ax[1].imshow(vnb.T, cmap='RdBu', vmin=-0.5, vmax=0.5, extent=[0,len(vnb)/5,0,360])
# ax[2].set_aspect(1/3)
# ax[1].set_aspect(1/3)
# plt.show()
    maxangle = np.argmax((curvs ** 3).sum(axis=0))
    if dir == '../amoeba/maxcurv/6_0' or dir == '../amoeba/maxcurv/7_0':
        maxangle += 180
    if i >= 4:
        maxangle += 70
    curvs = np.roll(curvs, -maxangle+curvs.shape[1]//2, axis=1)
    print(curvs.shape)
    print(maxangle)

    defv = 0
    maxv = 5/6
    maxv = 1/2
    minv = maxv - 2*(maxv - defv)
    # print('WARNING: extent may be wrong')
    # if i == 0:
    #     curvs = curvs[100*5:800*5,:]
    # else:
    #     curvs = curvs[-1000*5:,:]
    # curvs = curvs[200*5:,:]
    curvs = curvs[-1000*5:,:]
    if i == 0:
        curvs = curvs[:700*5,:]
    plot = ax.imshow(curvs.T, cmap='RdBu', vmin=minv, vmax=maxv, extent=[0,len(curvs)/5,0,360])
    # ax.set_aspect(1/3)
    # if i == 0:
    #     ax.set_aspect(1.25 * 700 / 1000)
    # else:
    #     ax.set_aspect(1.25)
    ax.set_aspect(1.2)
    if i == 0:
        ax.set_aspect(1.2/1000*700)
    ax.set_yticks(np.arange(0,360+1,90))
    if i >= len(sys.argv) - ncols - 1:
        ax.set_xlabel('Time (s)')
    else:
        pass
        # ax.set_xticklabels([])
    if i%ncols == 0:
        ax.set_ylabel('Angle (deg)')
    else:
        ax.set_yticklabels([])
    # ax.set_title('r_κ = ' + dir[-3:].replace('_', '.'))
    # ax.set_title('r_κ = ' + dir.split('/')[-1].replace('_', '.'))
    # ax.set_title(dir.split('/')[-1])
    # fig.colorbar(plot, fraction=0.047 * (curvs.shape[1]/curvs.shape[0]))
    # fig.colorbar(plot, fraction=0.2, label="Curvature (1/μm)")

# diffs = np.diff(curvs, axis=0)
# zipped = np.dstack((curvs[:-1], diffs))
# zipped = zipped.reshape(-1, zipped.shape[-1])
# print(curvs.shape)

# plt.scatter(curvs[100:-1].flatten(), (curvs[101:] - curvs[100:-1]).flatten(), s=0.2)
# plt.scatter(curvs[:-1].flatten(), diffs.flatten(), s=0.02)
# plt.plot([-0.05,0.15], [-0.0,0.0], ls='dashed', c='gray')

# increasing, bins = np.histogram(zipped[zipped[:,1] > 0][:,0], bins=30, range=(-0.05,0.15))
# decreasing, bins = np.histogram(zipped[zipped[:,1] < 0][:,0], bins=bins)
# print(increasing)
# print(decreasing)
# proportion = increasing / (increasing + decreasing)
# std = np.sqrt((decreasing * proportion**2 + increasing * (1-proportion)**2)/(increasing+decreasing))
# ci = 1.96 * std / np.sqrt(increasing+decreasing)  # 1.96: 95% CI
# print(proportion)
# plt.plot([-0.05,0.15], np.ones(2) * 0.5, ls='dashed', c='gray')
# idir = 0
# plt.plot((bins[:-1] + bins[1:])/2, proportion, c=f'C{idir}', label=dir)
# plt.plot((bins[:-1] + bins[1:])/2, proportion+ci, ls='dashed', c=f'C{idir}')
# plt.plot((bins[:-1] + bins[1:])/2, proportion-ci, ls='dashed', c=f'C{idir}')

# plt.legend()
# plt.show()

fig.subplots_adjust(right=1.1)
# cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
cbar_ax = fig.add_axes([0.85, 0.375, 0.02, 0.5])
plt.colorbar(plot, label="Curvature (1/μm)", cax=cbar_ax)
plt.savefig('kymograph.png', dpi=600, bbox_inches='tight')
# plt.savefig('kymograph.png', dpi=500)
# plt.show()

