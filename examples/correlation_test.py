from vae_tools.evaluation import corr_pc, corr_sc, corr_tc, corr_cca
import numpy as np

def correlation(u, v):
    print("PC: ", corr_pc(u, v))
    print("SC: ", corr_sc(u, v))
    # print("TC: ", corr_tc(u, v))
    print("CCA: ", corr_cca(u, v))

#%% random data
print("Random data")
num = 100
u = np.random.rand(num,2)
v = np.random.rand(num,2)
correlation(u, v)

#%% Prefect correlation
print("Prefect correlation")
u = np.asarray([[0., 0., -1.], [0,0.,0.], [0.,0.,1.], [0.,0.,2.]])
v = np.asarray([[1.,0.], [0., 0.], [-1., 0.], [-2., 0.]])
correlation(u, v)

#%% Handcrafted data
print("Handcrafted data")
u = np.asarray([[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [3.,5.,4.]])
v = np.asarray([[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]])
correlation(u, v)

#%% circular data
print("Circular data")
num = 100
tmp = np.concatenate((np.sin(np.linspace(0, 2*np.pi, num, False))[np.newaxis,:],
                      np.cos(np.linspace(0, 2*np.pi, num, False))[np.newaxis,:]), axis=0).T
u = 2 * tmp
v = 1 * tmp
correlation(u, v)

#%% wavy data

# This example shows a strange behaviour on CCA. While all other correlation metrics show pretty
# invariant correlation coefficient related to the number of samples, CCA does so heavily.

def get_data(num):
    u = np.concatenate((np.sin(np.linspace(0, 4 * np.pi, num, False))[np.newaxis, :],
                        np.linspace(0, 4 * np.pi, num, False)[np.newaxis, :]), axis=0).T
    v = np.concatenate((np.sin(np.linspace(0, 2 * np.pi, num, False))[np.newaxis, :],
                        np.cos(np.linspace(0, 2 * np.pi, num, False))[np.newaxis, :]), axis=0).T
    return u, v

print("Wavy data: 10")
u, v = get_data(10)
correlation(u, v)

print("Wavy data: 10")
u, v = get_data(100)
correlation(u, v)



#from scipy.spatial.distance import directed_hausdorff
#print(max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0]))
#print(directed_hausdorff(u, v))


