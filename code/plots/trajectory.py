
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

for i,dir in enumerate(sys.argv[1:]):
    shift = read_file(dir+f'/shift.txt')
    centroid = read_file(dir+f'/centroid.txt')
    dist = 0

    loc = (shift + centroid) * 0.15
    loc = loc[500:]
    loc[:,0] += (i%3)*100
    loc[:,1] += (i//3)*100
    cmap = plt.get_cmap('autumn')
    for i in range(loc.shape[0]-1):
        plt.plot(loc[i:i+2,0], -loc[i:i+2,1], c=cmap(i/loc.shape[0]))
        dist += np.sqrt((loc[i+1,0]-loc[i,0])**2 + (loc[i+1,1]-loc[i,1])**2)
    plt.plot(loc[0,0], -loc[0,1], 'o', c='k')
    plt.annotate(dir.split('/')[-1], (loc[0,0], -loc[0,1]))
    total_displ = np.sqrt((loc[-1,0]-loc[0,0])**2 + (loc[-1,1]-loc[0,1])**2)
    print(dir, dist, dist/(loc.shape[0]/5), total_displ/dist)

# plt.plot(loc[:,0], loc[:,1], color=np.linspace(0,1,loc.shape[0]))
plt.xlabel("Distance (µm)")
plt.ylabel("Distance (µm)")
plt.savefig('trajectory.png')
# plt.show()

