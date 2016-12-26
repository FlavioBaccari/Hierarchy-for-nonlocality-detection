from __future__ import print_function, division
from math import sqrt
from qutip import tensor, basis, expect, qeye, ket2dm
from ncpol2sdpa import generate_variables


def generate_commuting_measurements(party, label):
    """Genrates the list of symbolic variables representing the measurements
    for a given party. The variables are treated as commuting.

    :param party: configuration indicating the configuration of number m
                  of measurements and outcomes d for each measurement. It is a
                  list with m integers, each of them representing the number of
                  outcomes of the corresponding  measurement.
    :type party: list of int
    :param label: label to represent the given party
    :type label: str

    :returns: list of sympy.core.symbol.Symbol
    """
    measurements = []
    for i, p in enumerate(party):
        measurements.append(generate_variables(label + '%s' % i, p - 1,
                                               hermitian=True))
    return measurements


def get_moment(state, lambda_, operators, first, second=None, third=None,
               fourth=None):
    """Computes the value of the correlator
    <M_{x_i}^{i} M_{x_j}^{j} M_{x_k}^{k} M_{x_l}^{l} for the given 4-body
    reduced state of the N-partite state, the measurement operators and the
    noise parameter lambda_.

    :param state: 4-body reduced state corresponding the the parties i,j,k,l.
    :type state: qutip.qobj.Qobj
    :param lambda_: parameter representing the white noise added to the state.
    :type lambda_: sympy.core.symbol.Symbol
    :param operators: list of measurement operators for all the parties.
    :type operators: list of qutip.qobj.Qobj
    :param first: indices [i,x_i] representing the first party and
                  corresponding measurement choice entering in the correlator.
    :type first: list of int
    :param second: indices [j,x_j] representing the second party and
                   corresponding measurement choice entering in the correlator.
                   Optional, in case one is interested to measure a 1-body
                   correlator.
    :type second: list of int
    :param third: indices [k,x_k] representing the third party and
                  corresponding measurement choice entering in the correlator.
                  Optional, in case one is interested to measure a 2-body
                  correlator.
    :type third: list of int
    :param fourth: indices [l,x_l] representing the fourth party and
                   corresponding measurement choice entering in the correlator.
                   Optional, in case one is interested to measure a 3-body
                   correlator.
    :type fourth: list of int

    :returns: float value of the request correlator in the noiseless case
              or sympy.core.add.Add in the noisy case
    """

    projectors = [qeye(2) for _ in range(4)]
    projectors[0] = operators[first[0]][first[1]]
    if second is not None:
        projectors[1] = operators[second[0]][second[1]]
    if third is not None:
        projectors[2] = operators[third[0]][third[1]]
    if fourth is not None:
        projectors[3] = operators[fourth[0]][fourth[1]]
    noise = tensor([qeye(2) for _ in range(4)]) / 16
    return (1 - lambda_) * expect(tensor(projectors), state) + \
        lambda_ * expect(tensor(projectors), noise)


def get_moment_constraints(N, state, measurements, operators, lambda_=None,
                           order=None):
    """Generates the list of moment equalities substitution for the sdp
    relaxation at level 2, that is, the values of the few-body correlators for
    order up to four for a given state and measurement operators. Optional:
    adding the noise parameter lambda_ and fixing the maximal order of
    correlator that one wants to compute. It only works for symmetric states.

    :param N: number of particles.
    :type N: int
    :param state: the reduced 4-body quantum state. It is assumed to be the.
                  same for any fourtuple of particles.
    :type solver: qutip.qobj.Qobj
    :param measurements: list of commuting variables representing the local
                         measurements in the given scenario.
    :type measurements: list of sympy.core.symbol.Symbol
    :param operators: list of the measurements operators performed by the
                      parties.
    :type operators: list of qutip.qobj.Qobj
    :param lambda: symbolic variable corresponding to the amount of white noise
                   added to the state. Optional, in case one is interested in
                   computing the constraints for the noiseless case.
    :type lambda_: sympy.core.symbol.Symbol
    :param order: maximal order of the correlators that will be computed.
    :type order: int

    :returns: moments: list of sympy.core.add.Add

    """
    moments = {}

    # setting the order to 4 unless specified
    if order is None:

        order = 4

    # setting the noise parameter to zero unless specified
    if lambda_ is None:

        lambda_ = 0

    # generating all the one-body expectation values
    for k1, party1 in enumerate(measurements):
        for j1, measurement1 in enumerate(party1):
            for operator1 in measurement1:
                moments[operator1] = get_moment(state, lambda_, operators,
                                                (k1, j1))

    # generating all the two-body correlators
    if order >= 2:

        for k1, party1 in enumerate(measurements):
            for k2, party2 in enumerate(measurements[k1 + 1:], start=k1 + 1):
                for j1, measurement1 in enumerate(party1):
                    for operator1 in measurement1:

                        for j2, measurement2 in enumerate(party2):
                            for operator2 in measurement2:
                                moment = get_moment(state, lambda_, operators,
                                                    (k1, j1), (k2, j2))
                                moments[operator1 * operator2] = moment

    # generating all the three-body correlators
    if order >= 3:

        for k1, party1 in enumerate(measurements):
            for k2, party2 in enumerate(measurements[k1 + 1:], start=k1 + 1):
                for k3, party3 in enumerate(measurements[k2 + 1:], start=k2 + 1):
                    for j1, measurement1 in enumerate(party1):
                        for operator1 in measurement1:

                            for j2, measurement2 in enumerate(party2):
                                for operator2 in measurement2:

                                    for j3, measurement3 in enumerate(party3):
                                        for operator3 in measurement3:
                                            moment = get_moment(state, lambda_, operators, (k1, j1),
                                                                (k2, j2), (k3, j3))
                                            moments[
                                                operator1 * operator2 * operator3] = moment

    # generating all the four-body correlators
    if order >= 4:
        for k1, party1 in enumerate(measurements):
            for k2, party2 in enumerate(measurements[k1 + 1:], start=k1 + 1):
                for k3, party3 in enumerate(measurements[k2 + 1:], start=k2 + 1):
                    for k4, party4 in enumerate(measurements[k3 + 1:], start=k3 + 1):
                        for j1, measurement1 in enumerate(party1):
                            for operator1 in measurement1:

                                for j2, measurement2 in enumerate(party2):
                                    for operator2 in measurement2:

                                        for j3, measurement3 in enumerate(party3):
                                            for operator3 in measurement3:

                                                for j4, measurement4 in enumerate(party4):
                                                    for operator4 in measurement4:
                                                        moment = get_moment(state, lambda_, operators,
                                                                            (k1, j1), (k2, j2),
                                                                            (k3, j3), (k4, j4))
                                                        moments[operator1 * operator2 *
                                                                operator3 * operator4] = moment

    return moments


def get_W_state(N):
    """Generates the density matrix for the N-partite W state.

    :param N: number of parties.
    :type N: int

    :returns: the W density matrix as a qutip.qobj.Qobj
    """
    state = tensor([basis(2, 1)] + [basis(2, 0) for _ in range(N - 1)])
    for i in range(1, N):
        components = [basis(2, 0) for _ in range(N)]
        components[i] = basis(2, 1)
        state += tensor(components)
    return 1. / sqrt(N) * state


def get_W_reduced(N):
    """Generates the reduced four-body state for the N-partite W state. Since
    the W state is symmetric, it is independent of the choice of the four
    parties that one considers.

    :param N: number of parties for the global state.
    :type N: int

    :returns: the reduced state as a qutip.qobj.Qobj
    """
    w = ket2dm(get_W_state(4))
    rest = ket2dm(tensor([basis(2, 0) for _ in range(4)]))

    return 4. / N * w + (N - 4.) / N * rest


def get_GHZ_reduced(N):
    """Generates the reduced four-body state for the N-partite GHZ state. Since
    the GHZ state is symmetric, it is independent of the choice of the four
    parties that one considers.

    :param N: number of parties for the global state,
    :type N: int

    :returns: the reduced state as a qutip.qobj.Qobj
    """
    zero = tensor([basis(2, 0) for _ in range(N)])
    one = tensor([basis(2, 1) for _ in range(N)])
    return 1 / 2 * (ket2dm(zero) + ket2dm(one))


def get_fullmeasurement(indices, measurements):
    """Generates the monomial corresponding to a full-body correlator, with the
    measurement choices indicated by the indices list

    :param indices: list of measurement choices for each party, ranging from
                    0 to m-1.
    :type indices: list of int
    :param measurements: list of the symbolic variables representing the
                         measurements.
    :type measurements: list of sympy.core.symbol.Symbol

    :returns: monomial repreprenting the full-body correlator as a
             sympy.core.mul.Mul
    """

    operator = measurements[0][int(indices[0])][0]
    for i in range(1, len(indices)):
        operator *= measurements[i][int(indices[i])][0]
    return operator
