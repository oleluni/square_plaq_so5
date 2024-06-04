import numpy as np
import tools_custom.tools_build_opers as tools
import constants as const

def get_ravel_from_tensor(indices, dimensions):
    """
    Formulas to compute the flat index are presented below.
    In case counting starts from 1:
    c(i_1, ..., i_N) = \sum_{j=1}^{N} (i_{N+1 - j} - 1 ) \prod_{k=1}^{N-1} D_k

    In case counting starts from 0 (the useful one):
    c(i_0, ..., i_{N-1}) = \sum_{N_D-1}^{0} i_j * \prod_{k=0}^{j-1} D_{k}

    The algorythm uses F-ordering and is a custom adaptation of np.ravel_multi_index() .

    Args:
        indices (array): array of usual matrix indices i_j, such that:
        0 <= i_j <= D_j - 1 (for any j)

        dimensions (tuple): shape tuple of a tensor

    Returns:
    count (int): flat array index (checkerboard index)
    """

    N_dims = len(dimensions)

    # check if the input indices are within the corresponding dimensions
    for i, idx in enumerate(indices):
        if idx < 0 or idx >= dimensions[i]:
            raise ValueError(f'Index {idx} is out of bounds for dimension {i + 1}.')

    result = 0

    # calculate the flat index (start at 0) according to the formula above
    for j in range(N_dims - 1, -1, -1):
        result += (indices[j]) * np.prod(dimensions[:j])

    return result

def get_tensor_from_ravel(c: int, dimensions: tuple):
    """
    For a tensor of given shape, return the tensor indices array tensor_indices,
    corresponding to a given flat index c.

    Args:
        c (int): flat index, 0 <= c <= c_max - 1,
        where c_max = np.prod(dimensions)
        dimensions (tuple): shape tuple of the tensor

    Returns:
        tensor_indices (array): array of tensor indices corresponding to the
        value of c (flat index)
    """
    N_dims = len(dimensions)
    tensor_indices = np.zeros(N_dims)
    for j in range(N_dims):
        tensor_indices[j] = (c // np.prod(dimensions[:j])) % dimensions[j]

    return tensor_indices

def get_ravel_from_irrep(j, mL, mR):
    """
    Get checkerboard (flat) index for a given state (j, mL, mR).

    Formula to get matrix indices (i1, i2) from (j, mL, mR) labels:
    for -j <= mL, mR <= j
    i1 = (j - mL)
    i2 = (j - mR)

    Args:
        j (int or half-int): 0 <= j <= q
        mL: -j <= mL <= j
        mR: -j <= mR <= j

    Returns:
        i (int): flat index for a given state (j, mL, mR)
    """

    # sum previous counts up to (j - 1/2)
    count = tools.get_Nq(j-0.5)

    # get flat index i directly from (m1, m2) pair of indices
    i = (2*j + 1) * (j - mR) + j - mL

    # shift the indices by the counts from previous j's
    i += count

    return int(i)

def get_irrep_from_ravel(c: int):
    # j defines the dimension
    # use k = 2*j to have integers
    k, num_states = 0, 1
    while (c//num_states != 0):
        k += 1
        num_states += (k + 1) ** 2
    # assign j, the first quantum number
    j = k / 2

    # f is the flat index within the j-irrep
    # subtracting to get rid of the counts for previous j's
    f = c - tools.get_Nq(j-0.5)
    dim = 2*j + 1
    # i0, i1 = (f % dim), (f // dim)
    i0, i1 = map(int, get_tensor_from_ravel(f, (dim, dim)))
    mL, mR = j - i0, j - i1

    return j, mL, mR


def get_irrep_from_index_h4(c: int, q=const.TRUNCATION, n_sites=const.PLAQUETTE_SITES_NUMBER) -> tuple:
    """
    For the case of the tensor-product Hilbert space define function that returns the corresponding
    tensor-product irrep (tuple of 3 * n_sites = 12 entries) from the given ordinal index of the given state.

    Use the following numbering convention - for site # 1 (out of 4) define a certain mathematical expression, through
    the index of the state within one site can be inferred - i1.

    Assign N_q (5 for the q=1/2 case) states at each of the sites. When tensor product is taken over them to obtain the
    tensor-product Hilbert space, there should be certain intrinsic enumeration. The ordinal index of the large Hilbert
    space is "c". States at each site are also numbered i.e.: i1 - number of a state at the first link.

    The objective is to first put together the following tuple (i1, i2, i3, i4)
    Secondly, to get the irrep from this one has to apply function get_irrep_from_ravel() to each i1, i2, i3, i4.
    And lastly, this has to be put into one big 3 * 4 = 12 sized tuple.
    Args:
        c: ravel index of the tensor-product Hilbert space
        q: truncation
        n_sites: number of states in the plaquette, taken to be 4

    Returns:
        h4_irrep (tuple): irrep of the state in the Hilbert space, obtained from its ordinal number

    """
    Nq = tools.get_Nq(q=q)

    if c >= Nq**n_sites:
        raise ValueError("Provided index is larger than the Hilbert space size. Try increasing truncation 'q'.")

    i4 = c % Nq
    i3 = (c // Nq) % Nq
    i2 = ((c // Nq) // Nq) % Nq
    i1 = (((c // Nq) // Nq) // Nq) % Nq

    h4_irrep = tuple(map(get_irrep_from_ravel, (i1, i2, i3, i4)))

    # define a lambda function to flatten the tuple
    flatten_tuple = lambda nested_tuple: tuple(item for inner_tuple in nested_tuple for item in inner_tuple)

    h4_irrep_flat = flatten_tuple(h4_irrep)
    return h4_irrep_flat


# result = get_irrep_from_index_h4(c=27483, q=1)
# print(result)



# for c in range(4):
#     print(get_irrep_from_ravel(c))

# for c in range(1, 31):
#     print(c,'\t->',get_irrep_from_ravel(c), end='\t')
#     if get_irrep_from_ravel(c)[0] > get_irrep_from_ravel(c-1)[0] and c != 0:
#         print('\t')


# for c in range(tools.get_Nq(q=1)):
#     get_irrep_from_ravel(c)

# print(get_irrep_from_ravel(11))
# _test get_tensor_from_ravel()
# dims = (3, 2)
# c_max = np.prod(dims)
# for c in range(c_max):
#  print(c,'->',get_tensor_from_ravel(c, dims))

# testing custom_ravel()
#
# dims = (2, 2, 2)
# for i in range(dims[0]):
#     for j in range(dims[1]):
#         for k in range(dims[2]):
#             print(f"[{i}{j}{k}] -> ",my_ravel([i, j, k], dims), end=" ")
#         print("\n")
#
# for i in range(dims[0]):
#     for j in range(dims[1]):
#         for k in range(dims[2]):
#             print(f"[{i}{j}{k}] -> ",my_ravel([i, j, k], dims), end=" ")
#         print("\n")
