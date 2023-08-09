"""
Calculates the plaquette value
"""
from time import time
from typing import Tuple
import numpy as np  # type: ignore


def multiply_mat_mat(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """
    Multiple left by right. Assumes 3x3 (colour) (complex) matrices
    """
    mat_mat = np.empty([3, 3], dtype="complex")
    # do the maths for the colour matrices
    mat_mat[0, 0] = (
        left[0, 0] * right[0, 0] + left[0, 1] * right[1, 0] + left[0, 2] * right[2, 0]
    )
    mat_mat[1, 0] = (
        left[1, 0] * right[0, 0] + left[1, 1] * right[1, 0] + left[1, 2] * right[2, 0]
    )
    mat_mat[2, 0] = (
        left[2, 0] * right[0, 0] + left[2, 1] * right[1, 0] + left[2, 2] * right[2, 0]
    )
    # second index
    mat_mat[0, 1] = (
        left[0, 0] * right[0, 1] + left[0, 1] * right[1, 1] + left[0, 2] * right[2, 1]
    )
    mat_mat[1, 1] = (
        left[1, 0] * right[0, 1] + left[1, 1] * right[1, 1] + left[1, 2] * right[2, 1]
    )
    mat_mat[2, 1] = (
        left[2, 0] * right[0, 1] + left[2, 1] * right[1, 1] + left[2, 2] * right[2, 1]
    )
    # third index
    mat_mat[0, 2] = (
        left[0, 0] * right[0, 2] + left[0, 1] * right[1, 2] + left[0, 2] * right[2, 2]
    )
    mat_mat[1, 2] = (
        left[1, 0] * right[0, 2] + left[1, 1] * right[1, 2] + left[1, 2] * right[2, 2]
    )
    mat_mat[2, 2] = (
        left[2, 0] * right[0, 2] + left[2, 1] * right[1, 2] + left[2, 2] * right[2, 2]
    )
    return mat_mat


def multiply_matdag_matdag(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """
    #!Multiplies two (3,3) complex matrices together. Takes conjugate
    Does (left*right)^dagger
    """
    mat_mat = np.empty([3, 3], dtype="complex")
    # take transpose manually
    mat_mat[0, 0] = np.conj(
        left[0, 0] * right[0, 0] + left[1, 0] * right[0, 1] + left[2, 0] * right[0, 2]
    )
    mat_mat[1, 0] = np.conj(
        left[0, 1] * right[0, 0] + left[1, 1] * right[0, 1] + left[2, 1] * right[0, 2]
    )
    mat_mat[2, 0] = np.conj(
        left[0, 2] * right[0, 0] + left[1, 2] * right[0, 1] + left[2, 2] * right[0, 2]
    )
    # but take conjugate using np
    mat_mat[0, 1] = np.conj(
        left[0, 0] * right[1, 0] + left[1, 0] * right[1, 1] + left[2, 0] * right[1, 2]
    )
    mat_mat[1, 1] = np.conj(
        left[0, 1] * right[1, 0] + left[1, 1] * right[1, 1] + left[2, 1] * right[1, 2]
    )
    mat_mat[2, 1] = np.conj(
        left[0, 2] * right[1, 0] + left[1, 2] * right[1, 1] + left[2, 2] * right[1, 2]
    )
    # last index
    mat_mat[0, 2] = np.conj(
        left[0, 0] * right[2, 0] + left[1, 0] * right[2, 1] + left[2, 0] * right[2, 2]
    )
    mat_mat[1, 2] = np.conj(
        left[0, 1] * right[2, 0] + left[1, 1] * right[2, 1] + left[2, 1] * right[2, 2]
    )
    mat_mat[2, 2] = np.conj(
        left[0, 2] * right[2, 0] + left[1, 2] * right[2, 1] + left[2, 2] * right[2, 2]
    )
    return mat_mat


def real_trace_mult_mat_mat(left: np.ndarray, right: np.ndarray) -> float:
    """
    # !Takes the real trace of (3,3) complex numbers left, right multiplied together
    Tr(left*right)
    """
    tr_mat_mat = np.real(
        left[0, 0] * right[0, 0]
        + left[0, 1] * right[1, 0]
        + left[0, 2] * right[2, 0]
        + left[1, 0] * right[0, 1]  # noqa: W504
        + left[1, 1] * right[1, 1]
        + left[1, 2] * right[2, 1]
        + left[2, 0] * right[0, 2]  # noqa: W504
        + left[2, 1] * right[1, 2]
        + left[2, 2] * right[2, 2]
    )
    return tr_mat_mat


def plaquette(
    data: np.ndarray, mu_start: int = 0, mu_end: int = 4, nu_end: int = 4
) -> Tuple[float, int, float, float]:
    """
    Calculates the plaquette over mu_start to mu_end
    data is [nt, nx, ny, nz, mu, colour, colour] complex
    the plaquette over all lattice is mu_start=0, mu_end=4
    the spatial plaquette is mu_start=1, mu_end=4, nu_end=4
    the temporal plaquette is mu_start=0, mu_end=1, nu_end=4
    returns the sum of plaquettes, number of plaquettes measured,
    the average plaquette and the time taken to calculate it
    """
    start = time()
    shape = np.shape(data)
    # hold the sum
    sum_tr_plaq = 0.0
    # hold the number measured
    num_plaq = 0
    for mu in range(mu_start, mu_end):
        mu_coord = [0] * 4
        # This is the shift in mu
        mu_coord[mu] = 1
        for nu in range(mu + 1, nu_end):
            nu_coord = [0] * 4
            # This is the shift in nu
            nu_coord[nu] = 1
            # loop over all sites
            for nx in range(0, shape[1]):
                for ny in range(0, shape[2]):
                    for nz in range(0, shape[3]):
                        for nt in range(0, shape[0]):
                            # U_mu(x)
                            coord_base = np.asarray([nt, nx, ny, nz])
                            coord = coord_base
                            umu_x = data[
                                coord[0], coord[1], coord[2], coord[3], mu, :, :
                            ]
                            # U_nu(x + amu)
                            coord = coord_base + mu_coord
                            # respect periodic boundary conditions
                            for cc, co in enumerate(coord):
                                if co >= shape[cc]:
                                    coord[cc] = 0
                            unu_xmu = data[
                                coord[0], coord[1], coord[2], coord[3], nu, :, :
                            ]
                            # U_mu(x + anu)
                            coord = coord_base + nu_coord
                            for cc, co in enumerate(coord):
                                if co >= shape[cc]:
                                    coord[cc] = 0
                            umu_xnu = data[
                                coord[0], coord[1], coord[2], coord[3], mu, :, :
                            ]
                            # U_nu(x)
                            coord = coord_base
                            unu_x = data[
                                coord[0], coord[1], coord[2], coord[3], nu, :, :
                            ]
                            # Multiply bottom, right together
                            umu_unu = multiply_mat_mat(umu_x, unu_xmu)
                            # Multiply left, top together, take dagger
                            umudag_unudag = multiply_matdag_matdag(umu_xnu, unu_x)
                            # multiply two halves together, take trace
                            plaq = real_trace_mult_mat_mat(umu_unu, umudag_unudag)
                            sum_tr_plaq = sum_tr_plaq + plaq
                            num_plaq = num_plaq + 1
    end = time()
    return sum_tr_plaq, num_plaq, sum_tr_plaq / float(num_plaq), end - start
