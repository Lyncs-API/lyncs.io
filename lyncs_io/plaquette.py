"""
Calculates the plaquette value
"""
import numpy as np  # type: ignore
from time import time
from typing import Tuple


def MultiplyMatMat(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """
    Multiple left by right. Assumes 3x3 (colour) (complex) matrices
    """
    MM = np.empty([3, 3], dtype="complex")
    # do the maths for the colour matrices
    MM[0, 0] = (
        left[0, 0] * right[0, 0] + left[0, 1] * right[1, 0] + left[0, 2] * right[2, 0]
    )
    MM[1, 0] = (
        left[1, 0] * right[0, 0] + left[1, 1] * right[1, 0] + left[1, 2] * right[2, 0]
    )
    MM[2, 0] = (
        left[2, 0] * right[0, 0] + left[2, 1] * right[1, 0] + left[2, 2] * right[2, 0]
    )
    # second index
    MM[0, 1] = (
        left[0, 0] * right[0, 1] + left[0, 1] * right[1, 1] + left[0, 2] * right[2, 1]
    )
    MM[1, 1] = (
        left[1, 0] * right[0, 1] + left[1, 1] * right[1, 1] + left[1, 2] * right[2, 1]
    )
    MM[2, 1] = (
        left[2, 0] * right[0, 1] + left[2, 1] * right[1, 1] + left[2, 2] * right[2, 1]
    )
    # third index
    MM[0, 2] = (
        left[0, 0] * right[0, 2] + left[0, 1] * right[1, 2] + left[0, 2] * right[2, 2]
    )
    MM[1, 2] = (
        left[1, 0] * right[0, 2] + left[1, 1] * right[1, 2] + left[1, 2] * right[2, 2]
    )
    MM[2, 2] = (
        left[2, 0] * right[0, 2] + left[2, 1] * right[1, 2] + left[2, 2] * right[2, 2]
    )
    return MM


def MultiplyMatdagMatdag(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """
    #!Multiplies two (3,3) complex matrices together. Takes conjugate
    Does (left*right)^dagger
    """
    MM = np.empty([3, 3], dtype="complex")
    # take transpose manually
    MM[0, 0] = np.conj(
        left[0, 0] * right[0, 0] + left[1, 0] * right[0, 1] + left[2, 0] * right[0, 2]
    )
    MM[1, 0] = np.conj(
        left[0, 1] * right[0, 0] + left[1, 1] * right[0, 1] + left[2, 1] * right[0, 2]
    )
    MM[2, 0] = np.conj(
        left[0, 2] * right[0, 0] + left[1, 2] * right[0, 1] + left[2, 2] * right[0, 2]
    )
    # but take conjugate using np
    MM[0, 1] = np.conj(
        left[0, 0] * right[1, 0] + left[1, 0] * right[1, 1] + left[2, 0] * right[1, 2]
    )
    MM[1, 1] = np.conj(
        left[0, 1] * right[1, 0] + left[1, 1] * right[1, 1] + left[2, 1] * right[1, 2]
    )
    MM[2, 1] = np.conj(
        left[0, 2] * right[1, 0] + left[1, 2] * right[1, 1] + left[2, 2] * right[1, 2]
    )
    # last index
    MM[0, 2] = np.conj(
        left[0, 0] * right[2, 0] + left[1, 0] * right[2, 1] + left[2, 0] * right[2, 2]
    )
    MM[1, 2] = np.conj(
        left[0, 1] * right[2, 0] + left[1, 1] * right[2, 1] + left[2, 1] * right[2, 2]
    )
    MM[2, 2] = np.conj(
        left[0, 2] * right[2, 0] + left[1, 2] * right[2, 1] + left[2, 2] * right[2, 2]
    )
    return MM


def RealTraceMultMatMat(left: np.ndarray, right: np.ndarray) -> float:
    """
    # !Takes the real trace of (3,3) complex numbers left, right multiplied together
    Tr(left*right)
    """
    TrMM = np.real(
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
    return TrMM


def plaquette(
    data: np.ndarray, muStart: int = 0, muEnd: int = 4, nuEnd: int = 4
) -> Tuple[float, int, float, float]:
    """
    Calculates the plaquette over muStart to muEnd
    data is [nt, nx, ny, nz, mu, colour, colour] complex
    the plaquette over all lattice is muStart=0, muEnd=4
    the spatial plaquette is muStart=1, muEnd=4, nuEnd=4
    the temporal plaquette is muStart=0, muEnd=1, nuEnd=4
    returns the sum of plaquettes, number of plaquettes measured,
    the average plaquette and the time taken to calculate it
    """
    start = time()
    shape = np.shape(data)
    # hold the sum
    sumTrP = 0.0
    # hold the number measured
    nP = 0
    for mu in range(muStart, muEnd):
        muCoord = [0] * 4
        # This is the shift in mu
        muCoord[mu] = 1
        for nu in range(mu + 1, nuEnd):
            nuCoord = [0] * 4
            # This is the shift in nu
            nuCoord[nu] = 1
            # loop over all sites
            for nx in range(0, shape[1]):
                for ny in range(0, shape[2]):
                    for nz in range(0, shape[3]):
                        for nt in range(0, shape[0]):
                            # U_mu(x)
                            coordBase = np.asarray([nt, nx, ny, nz])
                            coord = coordBase
                            Umu_x = data[
                                coord[0], coord[1], coord[2], coord[3], mu, :, :
                            ]
                            # U_nu(x + amu)
                            coord = coordBase + muCoord
                            # respect periodic boundary conditions
                            for cc, co in enumerate(coord):
                                if co >= shape[cc]:
                                    coord[cc] = 0
                            Unu_xmu = data[
                                coord[0], coord[1], coord[2], coord[3], nu, :, :
                            ]
                            # U_mu(x + anu)
                            coord = coordBase + nuCoord
                            for cc, co in enumerate(coord):
                                if co >= shape[cc]:
                                    coord[cc] = 0
                            Umu_xnu = data[
                                coord[0], coord[1], coord[2], coord[3], mu, :, :
                            ]
                            # U_nu(x)
                            coord = coordBase
                            Unu_x = data[
                                coord[0], coord[1], coord[2], coord[3], nu, :, :
                            ]
                            # Multiply bottom, right together
                            UmuUnu = MultiplyMatMat(Umu_x, Unu_xmu)
                            # Multiply left, top together, take dagger
                            UmudagUnudag = MultiplyMatdagMatdag(Umu_xnu, Unu_x)
                            # multiply two halves together, take trace
                            P = RealTraceMultMatMat(UmuUnu, UmudagUnudag)
                            sumTrP = sumTrP + P
                            nP = nP + 1
    end = time()
    return sumTrP, nP, sumTrP / float(nP), end - start
