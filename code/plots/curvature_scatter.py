
import numpy as np
import sys
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import mean
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import binned_statistic

import matplotlib
matplotlib.rcParams.update({'font.size': 13})

dirs = sys.argv[1:]

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

for idir, dir in enumerate(dirs):
    # curvs = np.load(dir+'/menger_curv.npy')
    curvs = read_file(dir+f'/curv_histogram.txt')
    bct = read_file(dir+f'/bct_histogram.txt')
    print(bct.shape)
    # curvs = gaussian_filter(curvs, sigma=5)
    # plt.imshow(curvs)
    # plt.show()
    # print(curvs.shape)
    # for i in range(1,10):
    #     diff = curvs[i:,:] - curvs[:-i, :]
    #     print(i, np.abs(diff).mean())
    diff = np.diff(curvs, axis=0)

    filter = bct[:-1] >= 0.2
    mean_stat = binned_statistic(np.where(filter, curvs[:-1,:], np.nan).flatten(), diff.flatten(), statistic='mean', bins=np.arange(-2/6, 6/6+1e-5, 0.2/6))
    # plt.scatter(curvs[:-1,:].flatten(), diff.flatten(), s=0.001)
    xs = (mean_stat.bin_edges[:-1] + mean_stat.bin_edges[1:])/2
    # plt.plot(xs, mean_stat.statistic/5, c=f"C{idir}", ls='--')

    filter = bct[:-1] > 0.5
    mean_stat = binned_statistic(np.where(filter, curvs[:-1,:], np.nan).flatten(), diff.flatten(), statistic='mean', bins=np.arange(-2/6, 6/6+1e-5, 0.2/6))
    # plt.scatter(curvs[:-1,:].flatten(), diff.flatten(), s=0.001)
    xs = (mean_stat.bin_edges[:-1] + mean_stat.bin_edges[1:])/2

    if idir < len(dirs)-1:
        mk = float(dir[-3:].replace('_','.'))/6
        plt.plot(xs, mean_stat.statistic/5, c=f"C{idir}", label=f"κ_max = {mk*6:.0f}/6")
        plt.axvline(mk, c=f"C{idir}", ls=':')
    else:
        plt.plot(xs, mean_stat.statistic/5, c="k", label='Constant force')
        plt.axvline(0, c="k", ls=':')

plt.axes().set_aspect(18)
plt.grid()
plt.xlabel('κ (1/µm)')
plt.ylabel('dκ/dt ((µm s)^-1)')
plt.ylim([-0.03,0.03])
plt.xlim([-0.3,0.9])
plt.legend(loc='upper left', bbox_to_anchor=(0.8,1.1))
# plt.xticks(np.arange(-1/3,1,1/6))
plt.savefig("curvature_scatter.png", dpi=500, bbox_inches='tight')
plt.show()

