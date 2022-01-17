from __future__ import division, print_function

import numpy as np
import pylibxc
import pytest

from functionals import *

parameters = {}
parameterfile = open("constants.txt", "r")
for line in parameterfile:
    parameters[line.split("=")[0].strip()] = str(line.split("=")[1])


def run_PBEx_unpolar(rho0, gamma0, parameters):
    rho = symbols('rho', real=True, nonnegative=True)
    gamma = symbols('gamma', real=True, nonnegative=True)

    # unpolarized
    zk = PBEx(rho, gamma, parameters)
    E = zk * rho
    vrho = E.diff(rho)
    vsigma = E.diff(gamma)
    v2rho2 = vrho.diff(rho)
    v2rhosigma = vrho.diff(gamma)
    v2sigma2 = vsigma.diff(gamma)

    exprs = [zk, vrho, vsigma, v2rho2, v2rhosigma, v2sigma2]
    expr_names = ['zk', 'vrho', 'vsigma', 'v2rho2', 'v2rhosigma', 'v2sigma2']
    res = {}
    for expr, name in zip(exprs, expr_names):
        # rho_a = 2, gamma_aa = 3; rho_b = 2, gamma_bb = 3; gamma_ab=0
        res[name] = expr.evalf(subs={rho: rho0, gamma: gamma0})

    res_new = {}
    for k, v in res.items():
        res_new[k] = np.array([[v]], dtype=float)
    return res_new


def run_PBEx_polar(rho0, rho1, gamma0, gamma1, gamma2, parameters):
    # polarized
    rho_a = symbols('rho_a', real=True, nonnegative=True)
    rho_b = symbols('rho_b', real=True, nonnegative=True)
    gamma_aa = symbols('gamma_aa', real=True, nonnegative=True)
    gamma_ab = symbols('gamma_ab', real=True, nonnegative=True)
    gamma_bb = symbols('gamma_bb', real=True, nonnegative=True)
    zk_a = PBEx(2 * rho_a, 4 * gamma_aa, parameters)
    zk_b = PBEx(2 * rho_b, 4 * gamma_bb, parameters)
    E_a = 2 * rho_a * zk_a
    E_b = 2 * rho_b * zk_b
    E = 0.5 * (E_a + E_b)
    zk = E / (rho_a + rho_b)
    vrho_a = E.diff(rho_a)
    vrho_b = E.diff(rho_b)
    vsigma_aa = E.diff(gamma_aa)
    vsigma_ab = E.diff(gamma_ab)
    vsigma_bb = E.diff(gamma_bb)

    exprs = [zk, vrho_a, vrho_b, vsigma_aa, vsigma_ab, vsigma_bb]
    expr_names = ['zk', 'vrho0', 'vrho1', 'vsigma0', 'vsigma1', 'vsigma2']
    res = {}
    for expr, name in zip(exprs, expr_names):
        if np.isclose(rho0 + rho1, 0.0):
            res[name] = 0.0
            continue
        res[name] = expr.evalf(
            subs={rho_a: rho0, rho_b: rho1, gamma_aa: gamma0, gamma_ab: gamma1, gamma_bb: gamma2})

    res_new = {
        'zk': np.array([[res['zk']]], dtype=float),
        'vrho': np.array([[res['vrho0'], res['vrho1']]], dtype=float),
        'vsigma': np.array([[res['vsigma0'], res['vsigma1'], res['vsigma2']]], dtype=float)}

    return res_new


def run_PBEc_polar(rho0, rho1, gamma0, gamma1, gamma2, parameters):
    # polarized
    rho_a = symbols('rho_a', real=True, nonnegative=True)
    rho_b = symbols('rho_b', real=True, nonnegative=True)
    gamma_aa = symbols('gamma_aa', real=True, nonnegative=True)
    gamma_ab = symbols('gamma_ab', real=True, nonnegative=True)
    gamma_bb = symbols('gamma_bb', real=True, nonnegative=True)
    zk = PBEc(rho_a + rho_b, rho_a - rho_b, gamma_aa + 2 * gamma_ab + gamma_bb, parameters)
    E = (rho_a + rho_b) * zk
    vrho_a = E.diff(rho_a)
    vrho_b = E.diff(rho_b)
    vsigma_aa = E.diff(gamma_aa)
    vsigma_ab = E.diff(gamma_ab)
    vsigma_bb = E.diff(gamma_bb)

    exprs = [zk, vrho_a, vrho_b, vsigma_aa, vsigma_ab, vsigma_bb]
    expr_names = ['zk', 'vrho0', 'vrho1', 'vsigma0', 'vsigma1', 'vsigma2']
    res = {}
    for expr, name in zip(exprs, expr_names):
        res[name] = expr.evalf(
            subs={rho_a: rho0, rho_b: rho1, gamma_aa: gamma0, gamma_ab: gamma1, gamma_bb: gamma2})

    res_new = {
        'zk': np.array([[res['zk']]], dtype=float),
        'vrho': np.array([[res['vrho0'], res['vrho1']]], dtype=float),
        'vsigma': np.array([[res['vsigma0'], res['vsigma1'], res['vsigma2']]], dtype=float)}

    return res_new


def run_PBEc_unpolar(rho0, gamma0, parameters):
    # unpolarized
    rho = symbols('rho', real=True, nonnegative=True)
    gamma = symbols('gamma', real=True, nonnegative=True)
    zk = PBEc(rho, 0, gamma, parameters)
    E = rho * zk
    vrho = E.diff(rho)
    vsigma = E.diff(gamma)

    exprs = [zk, vrho, vsigma]
    expr_names = ['zk', 'vrho0', 'vsigma0']
    res = {}
    for expr, name in zip(exprs, expr_names):
        res[name] = expr.evalf(subs={rho: rho0, gamma: gamma0})

    res_new = {
        'zk': np.array([[res['zk']]], dtype=float),
        'vrho': np.array([[res['vrho0']]], dtype=float),
        'vsigma': np.array([[res['vsigma0']]], dtype=float)}

    return res_new


def run_TPSSx_unpolar(rho0, gamma0, tau0, lapl0, parameters):
    rho = symbols('rho', real=True, nonnegative=True)
    gamma = symbols('gamma', real=True, nonnegative=True)
    tau = symbols('tau', real=True, nonnegative=True)
    lapl = symbols('lapl', real=True, nonnegative=True)

    # unpolarized
    zk = TPSSx(rho, gamma, tau, lapl, parameters)
    E = zk * rho
    vrho = E.diff(rho)
    vsigma = E.diff(gamma)
    vtau = E.diff(tau)
    vlapl = E.diff(lapl)
    # v2rho2 = vrho.diff(rho)
    # v2rhosigma = vrho.diff(gamma)
    # v2sigma2 = vsigma.diff(gamma)

    exprs = [zk, vrho, vsigma, vtau, vlapl]  # v2rho2, v2rhosigma, v2sigma2]
    expr_names = ['zk', 'vrho', 'vsigma', 'vtau', 'vlapl']  # , 'v2rho2', 'v2rhosigma', 'v2sigma2']
    res = {}
    for expr, name in zip(exprs, expr_names):
        # rho_a = 2, gamma_aa = 3; rho_b = 2, gamma_bb = 3; gamma_ab=0
        res[name] = expr.evalf(subs={rho: rho0, gamma: gamma0, tau: tau0, vlapl: lapl0})

    res_new = {}
    for k, v in res.items():
        res_new[k] = np.array([[v]], dtype=float)
    return res_new


def get_libxc_ref(func_name, spin, rho0, rho1, gamma0, gamma1, gamma2, do_exc=True, do_vxc=True,
                  do_fxc=False, do_kxc=False, do_lxc=False):
    func = pylibxc.LibXCFunctional(func_name, spin)
    if spin == 'polarized':
        inp = {'rho': np.array([rho0, rho1]), 'sigma': np.array([gamma0, gamma1, gamma2])}
    else:
        inp = {'rho': np.array([rho0]), 'sigma': np.array([gamma0])}

    ret = func.compute(inp, do_exc=do_exc, do_vxc=do_vxc, do_fxc=do_fxc, do_kxc=do_kxc,
                       do_lxc=do_lxc)
    return ret


def run_PBEx_polar_libxc(rho0, rho1, gamma0, gamma1, gamma2):
    return get_libxc_ref('gga_x_pbe', 'polarized', rho0, rho1, gamma0, gamma1, gamma2)


def run_PBEx_unpolar_libxc(rho0, gamma0):
    return get_libxc_ref('gga_x_pbe', 'unpolarized', rho0, 0, gamma0, 0, 0, do_fxc=True)


def run_PBEc_polar_libxc(rho0, rho1, gamma0, gamma1, gamma2):
    return get_libxc_ref('gga_c_pbe', 'polarized', rho0, rho1, gamma0, gamma1, gamma2)


def run_PBEc_unpolar_libxc(rho0, gamma0):
    return get_libxc_ref('gga_c_pbe', 'unpolarized', rho0, 0, gamma0, 0, 0)


def run_TPSSx_unpolar_libxc(rho0, gamma0, tau0, lapl0):
    func = pylibxc.LibXCFunctional('mgga_x_tpss', 'unpolarized')
    inp = {'rho': np.array([rho0]), 'sigma': np.array([gamma0]),
           'tau': np.array([tau0]), 'lapl': np.array([lapl0])}
    ret = func.compute(inp)
    return ret


# unittests
@pytest.mark.parametrize('rho0', [1, 2, 3])
@pytest.mark.parametrize('gamma0', [0, 10, 20, 30])
def test_pbe_x(rho0, gamma0):
    res1 = run_PBEx_unpolar(2 * rho0, 4 * gamma0, parameters)
    res2 = run_PBEx_unpolar_libxc(2 * rho0, 4 * gamma0)
    for k, v in res1.items():
        assert v == pytest.approx(res2[k], abs=1e-5)


@pytest.mark.parametrize('rho0', [1, 2, 3])
@pytest.mark.parametrize('gamma0', [0, 10, 20, 30])
def test_pbe_c(rho0, gamma0):
    res1 = run_PBEc_unpolar(2 * rho0, 4 * gamma0, parameters)
    res2 = run_PBEc_unpolar_libxc(2 * rho0, 4 * gamma0)
    for k, v in res1.items():
        assert v == pytest.approx(res2[k], abs=1e-5)


# TPSS_x test failed.
@pytest.mark.parametrize('rho', [1])
@pytest.mark.parametrize('sigma', [20])
@pytest.mark.parametrize('tau', [2])
@pytest.mark.parametrize('lapl', [0])
def test_tpss_x(rho, sigma, tau, lapl):
    res1 = run_TPSSx_unpolar(rho, sigma, tau, lapl, parameters)
    res2 = run_TPSSx_unpolar_libxc(rho, sigma, tau, lapl)
    for k, v in res1.items():
        assert v == pytest.approx(res2[k], abs=1e-5)
