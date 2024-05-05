
import numpy as np
import matplotlib.pyplot as plt
import sys

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

fig, axs = plt.subplots(1, len(sys.argv[1:]))
titles = ['Directed', 'Rotating', 'Bursting', 'Static']

for idir, dir in enumerate(sys.argv[1:]):
    ax = axs[idir]
    shift = read_file(dir+f'/shift.txt')
    centroid = read_file(dir+f'/centroid.txt')
    dist = 0

    loc = (shift + centroid) * 0.15
    loc = loc[500:]

    minx = loc[:,0].min()
    maxx = loc[:,0].max()
    miny = loc[:,1].min()
    maxy = loc[:,1].max()
    span = max(maxx-minx, maxy-miny)
    cx = (maxx+minx)/2
    cy = (maxy+miny)/2

    cmap = plt.get_cmap('autumn')
    for i in range(loc.shape[0]-1):
        ax.plot(loc[i:i+2,0]-cx, -loc[i:i+2,1]+cy, c=cmap(i/loc.shape[0]))
        dist += np.sqrt((loc[i+1,0]-loc[i,0])**2 + (loc[i+1,1]-loc[i,1])**2)
    ax.plot(loc[0,0]-cx, -loc[0,1]+cy, 'o', c='k')
    # plt.annotate(dir.split('/')[-1], (loc[0,0], -loc[0,1]))
    total_displ = np.sqrt((loc[-1,0]-loc[0,0])**2 + (loc[-1,1]-loc[0,1])**2)
    print(dir, dist, dist/(loc.shape[0]/5), total_displ/dist, span)

    if idir in [0]:
        span = span * 0.6
    else:
        span = 15
    # span = 72
    ax.set_xlim(-span,span)
    ax.set_ylim(span,-span)
    ax.set_aspect(1/2)
    ax.set_xlabel("Distance (µm)")
    ax.set_title(titles[idir])

# plt.plot(loc[:,0], loc[:,1], color=np.linspace(0,1,loc.shape[0]))
# axs[len(axs)//2].set_xlabel("Distance (µm)")
axs[0].set_ylabel("Distance (µm)")
# plt.tight_layout()
# plt.savefig('trajectory.png', bbox_inches='tight', dpi=200)
plt.savefig('trajectory.png', dpi=200)
# plt.show()

