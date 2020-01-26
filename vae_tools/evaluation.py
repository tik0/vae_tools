import scipy.sparse as sparse
import scipy.stats as stats
import scipy.spatial as spatial
import numpy as np
from sklearn.cross_decomposition import CCA

def distance_matrix(x, y=None, metric='euclidean', exponent=1):
    """Pairwise distance between points in a set."""

    # sanity check
    def cure_dimesion(x):
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        if x.shape[0] == 1:
            x = x.T
        return x

    x = cure_dimesion(x)
    if y is not None:
        y = cure_dimesion(y)

    if metric == 'geodesic':
        # Geodesic distance is measured by the number of edges to
        # go from vertice i to vertice j
        if y is not None:
            raise Exception("Geodesic distance between graphs is not defined")
        if x.shape[1] == 1:  # Geodesic distance not defined on scalars, use rank_difference instead
            distances = distance_matrix(x, metric='rank_difference')
        else:
            # Build Delaunay triangulation
            tri = spatial.Delaunay(x, qhull_options='QJ')
            # construct dense graph
            num_verticies = x.shape[0]
            G = np.zeros((num_verticies, num_verticies), dtype=bool)
            # Check for connections between the verticies
            for x in np.arange(1, num_verticies):
                for y in np.arange(0, x):
                    if ((tri.simplices == x).any(axis=1) & (tri.simplices == y).any(axis=1)).any():
                        G[x, y] = True
            G |= G.T
            G_masked = np.ma.masked_values(G, 0)
            CS = sparse.csgraph.csgraph_from_masked(G_masked)
            # get the travel lenght between all nodes
            distances = sparse.csgraph.shortest_path(CS, method='auto', directed=False, unweighted=True)
    elif metric == 'rank_difference':
        if y is not None:
            raise Exception("Rank distance between graphs is not defined")
        ranks = rank(x)
        # Rank is 1 dimensional, so euclidean gives us the absolute distance between verticies
        distances = distance_matrix(x, metric='euclidean', exponent=1)
    else:
        if exponent != 1 and metric == 'euclidean':
            metric = 'sqeuclidean'

        if y is None:
            distances = spatial.distance.pdist(x, metric=metric)
            distances = spatial.distance.squareform(distances)
        else:
            distances = spatial.distance.cdist(x, y, metric=metric)

        if exponent != 1:
            distances **= exponent / 2

    return distances


def pairwise_distances(distance_matrix):
    """ Returns the upper triangular of a nxn distance matrix """
    n = distance_matrix.shape[0]
    return distance_matrix[np.triu_indices(n, k=1)]


def mean_pairwise_distance(distance_matrix):
    """ Calculate mean pairwise distance of the given distance nxn matrix

    @param dist: nxn distance matrix
    """
    return pairwise_distances(distance_matrix).mean()


def rank(x, use_scipy=True):
    """ calculates the rank of any given 1xn vector x"""
    # array = np.array([4,2,7,1])
    if use_scipy:  # allows tie handling
        ranks = stats.rankdata(x)
    else:
        temp = x.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(x))
    return ranks


def corr_test(pd_x, pd_y, correlation, corr_fun, permutations=10 ** 3, seed=0):
    """ Perform MC testing on the correlation """
    np.random.seed(seed)
    num_exceed = 0
    _pd_x = pd_x.copy()
    for idx in range(permutations):
        np.random.shuffle(_pd_x)
        c_shuffle, _ = corr_fun(_pd_x, pd_y, test=False)
        if c_shuffle > correlation:
            num_exceed += 1
    p_value = (num_exceed + 1) / (permutations + 1)
    return p_value

def corr_c(pd_x, pd_y, test=True, permutations=10 ** 3):
    """ Calculates the correlation coefficient """
    # TODO exact permutation analysis if all permutations are less than 10**6
    var_x = np.sum((pd_x - pd_x.mean()) ** 2)
    var_y = np.sum((pd_y - pd_y.mean()) ** 2)
    denom = np.sqrt(var_x * var_y)
    nom = np.sum((pd_x - pd_x.mean()) * (pd_y - pd_y.mean()))
    c = nom / denom
    # MC test
    if test:
        p_value = corr_test(pd_x, pd_y, c, corr_c, permutations)
    else:
        p_value = np.nan
    return c, p_value


def corr_pc(x, y):
    """Pearson distance correlation (PC) """
    distance_matrix_x, distance_matrix_y = distance_matrix(x), distance_matrix(y)
    pd_x = pairwise_distances(distance_matrix_x)
    pd_y = pairwise_distances(distance_matrix_y)
    return corr_c(pd_x, pd_y)


def corr_sc(x, y):
    """Spearman distance correlation (SC) """
    distance_matrix_x, distance_matrix_y = distance_matrix(x), distance_matrix(y)
    rank_x = rank(pairwise_distances(distance_matrix_x))
    rank_y = rank(pairwise_distances(distance_matrix_y))
    return corr_c(rank_x, rank_y)


def corr_tc(x, y):
    """Topological correlation (TC) """
    distance_matrix_x = distance_matrix(x, metric='geodesic')
    distance_matrix_y = distance_matrix(y, metric='geodesic')
    pd_x = pairwise_distances(distance_matrix_x)
    pd_y = pairwise_distances(distance_matrix_y)
    return corr_c(pd_x, pd_y)

def _corr_cca_cosine_sim(u_c, v_c, test = False):
    """ Calculates cosine similarity for CCA with interface for MC testing """
    axis = 1
    c = np.sum(u_c * v_c, axis) / (np.linalg.norm(u_c, ord=2, axis=axis) * np.linalg.norm(v_c, ord=2, axis=axis))
    return c.mean(), None

def corr_cca(u, v, test = True, n_components=2, permutations=10 ** 3):
    """ Canonical Cross Correlation CCA ref. https://arxiv.org/abs/1911.03393 """
    # u: [samples, sample_dimesion]
    # v: [samples, sample_dimesion]

    # get the canonical correlation
    cca = CCA(n_components)
    cca.fit(u, v)
    u_c, v_c = cca.transform(u, v)
    # center
    u_c -= u_c.mean()
    v_c -= v_c.mean()
    # get the mean correlation
    c, _ = _corr_cca_cosine_sim(u_c, v_c)
    # MC test
    p_value = np.nan
    if test:
        p_value = corr_test(u_c, v_c, c, _corr_cca_cosine_sim, permutations)
    return c, p_value
