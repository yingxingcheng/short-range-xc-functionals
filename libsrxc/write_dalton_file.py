from __future__ import print_function, division

import functionals_mu as mu_func
import functionals as func
import functionals_special as spec_func
from sympy import symbols, cse, fcode, Rational, sqrt, exp, ln, Max
import print_functional_to_DALTON as dalprint
import print_functional_to_DALTON_TPSSc_special_case as dalprint_spec

rho, rho_a, rho_b = symbols('rho rho_a rho_b', real=True, nonnegative=True)
sigma, sigma_aa, sigma_ab, sigma_bb = \
    symbols('sigma sigma_aa sigma_ab sigma_bb', real=True, nonnegative=True)
tau, tau_a, tau_b = symbols('tau tau_a tau_b', real=True, nonegative=True)
lapl, lapl_a, lapl_b = symbols('lapl lapl_a lapl_b', real=True, nonegative=True)
mu = symbols('mu', real=True, nonnegative=True)

# load parameters
parameters = {}
parameterfile = open("constants.txt", "r")
for line in parameterfile:
    parameters[line.split("=")[0].strip()] = str(line.split("=")[1])

# write_list = ["LDA_ERF_exchange","PBE_ERFGWS_exchange","TPSS_ERFGWS_exchange","PW92_ERF_correlation",
#              "PBE_ERFGWS_correlation","TPSS_ERFGWS_correlation","PBE_nomu_correlation","VWN5_ERF_correlation",
#              "wPBE_exchange","VWN5_nomu_correlation"]
# write_list = ["PW92_ERF_correlation", "VWN5_ERF_correlation"]
# write_list = ["PBE_ERFGWS_correlation"]
# write_list = ["LDA_ERF_exchange"]
# write_list = ["PBE_ERFGWS_exchange"]
# write_list = ["TPSS_ERFGWS_exchange", 'TPSS_ERFGWS_correlation']
write_list = ["TPSS_nomu_exchange"]
# write_list = ["TPSS_ERFGWS_correlation"]
# write_list = ["PW92_ERF_correlation"]
# write_list = ["VWN5_ERF_correlation"]
# write_list = ["PBE_nomu_correlation"]
# write_list = ["VWN5_nomu_correlation" ]


def get_diff_idx_from_order(diff_order):
    diff_idx = [[0]]
    if len(diff_order) > 1:
        diff_idx += [list(range(1, i + 1)) for i in diff_order[1:]]
    return diff_idx


def check_get_diff_idx_fron_order():
    diff_order = [1]
    assert get_diff_idx_from_order(diff_order) == [[0]]
    diff_order = [1, 5]
    assert get_diff_idx_from_order(diff_order) == [[0], [1, 2, 3, 4, 5]]
    diff_order = [1, 2, 3]
    assert get_diff_idx_from_order(diff_order) == [[0], [1, 2], [1, 2, 3]]


def get_derivative(E_tot_list, vars):
    nvar = len(vars)

    d1E_all_cases, d2E_all_cases = [], []
    for E_tot in E_tot_list:
        d1E_list = []
        d2E_list = []
        for i in range(nvar):
            print('v' + vars[i].name)
            d1E_list.append(E_tot.diff(vars[i]))
            for j in range(i + 1):
                if i == j:
                    print('v2' + vars[i].name + '2')
                else:
                    print('v2' + vars[j].name + vars[i].name)
                d2E_list.append(d1E_list[-1].diff(vars[j]))
        d1E_all_cases.append(d1E_list)
        d2E_all_cases.append(d2E_list)
    return d1E_all_cases, d2E_all_cases


if "LDA_ERF_exchange" in write_list:
    out_file = open("../srfunctionals/LDA_ERF_exchange.F", "w+")
    # ######################################################################
    # LDA exchange
    out_file.write("C SOURCES:\n")
    out_file.write(
        "C    Simone Paziani, Saverio Moroni, Paola Gori-Giorgi, and Giovanni B. Bachelet.\n")
    out_file.write(
        "C    Local-spin-densityfunctional for multideterminant density functional theory.\n")
    out_file.write("C    Physical Review B, 73(15), apr 2006.\n")
    out_file.write("\n")
    # ######################################################################
    funcs = [
        mu_func.LDAx_mu_case_1,
        mu_func.LDAx_mu_case_2,
        mu_func.LDAx_mu_case_3,
    ]

    fnames = ["LDA_ERF_case_1",
              "LDA_ERF_case_2",
              "LDA_ERF_case_3"]


    def get_code(spin='unpolarized', shortrange=True):
        description = ["Implemented by YingXing Cheng.\n"]

        if spin == 'unpolarized':
            vars = [rho]
            zk_list = [_func(rho, mu, parameters) for _func in funcs]
            E_tot_list = [_zk * rho for _zk in zk_list]
            input = [var.name for var in vars]
            outputs = [['Ea'], ['Ea', 'd1Ea'], ['Ea', 'd1Ea', 'd2Ea']]
            prefixs = ['ESRX', 'D1ESRX', 'D2ESRX']
        else:
            vars = [rho_a, rho_b]
            E_tot_list = [_func(rho_a * 2, mu, parameters) * rho_a
                          + _func(rho_b * 2, mu, parameters) * rho_b
                          for _func in funcs]
            zk_list = [E_tot / (rho_a + rho_b) for E_tot in E_tot_list]
            input = [var.name for var in vars]
            outputs = [['E'], ['E', 'd1E'], ['E', 'd1E', 'd2E']]
            prefixs = ['ESRX_SPIN', 'D1ESRX_SPIN', 'D2ESRX_SPIN']

        nbv1 = len(vars)
        nbv2 = (nbv1 + 1) * nbv1 // 2
        diff_orders = [[1], [1, nbv1], [1, nbv1, nbv2]]

        d1E_all_cases, d2E_all_cases = get_derivative(E_tot_list, vars)

        for order in range(3):
            prefix = prefixs[order]
            output = outputs[order]
            diff_order = diff_orders[order]
            diff_idx = get_diff_idx_from_order(diff_order)
            func_names = [prefix + "_" + _fn for _fn in fnames]

            for i, fn in enumerate(func_names):
                if order == 0:
                    kernel = [zk_list[i]]
                elif order == 1:
                    kernel = [zk_list[i]] + d1E_all_cases[i]
                elif order == 2:
                    kernel = [zk_list[i]] + d1E_all_cases[i] + d2E_all_cases[i]
                else:
                    raise RuntimeError('order should be less than 3!')
                dalprint.dalton_functional_printer(kernel, fn, input, output,
                                                   description=description, shortrange=shortrange,
                                                   diff_order=diff_order, diff_idx=diff_idx,
                                                   output_files=[out_file])


    get_code('unpolarized')
    get_code('polarized')
    out_file.close()

if "PBE_ERFGWS_exchange" in write_list:
    out_file = open("../srfunctionals/PBE_ERFGWS_exchange.F", "w+")
    # ######################################################################
    # PBE exchange
    out_file.write("C SOURCES:\n")
    out_file.write("C    Erich Goll,  Hans-Joachim Werner, and Hermann Stoll.\n")
    out_file.write(
        "C    A short-range gradient-corrected densityfunctional in long-range coupled-cluster calculations for rare gas dimers.\n")
    out_file.write("C    Physical Chemistry Chemical Physics, 7(23):3917, 2005.\n")
    out_file.write("\n")
    out_file.write(
        "C    Erich Goll, Hans-Joachim Werner, Hermann Stoll, Thierry Leininger, Paola Gori-Giorgi, and Andreas Savin.\n")
    out_file.write(
        "C    A short-range gradient-corrected spin density functional in combination with long-range coupled-cluster methods:  Application to alkali-metal rare-gas dimers.\n")
    out_file.write("C    Chemical Physics,329(1-3):276-282, oct 2006.\n")
    out_file.write("\n")
    # ######################################################################

    funcs = [
        mu_func.PBEx_mu_case_1,
        mu_func.PBEx_mu_case_2_1,
        mu_func.PBEx_mu_case_2_2,
        mu_func.PBEx_mu_case_2_3,
        mu_func.PBEx_mu_case_3,
    ]

    fnames = ["PBE_GWS_ERF_case_1",
              "PBE_GWS_ERF_case_2_1",
              "PBE_GWS_ERF_case_2_2",
              "PBE_GWS_ERF_case_2_3",
              "PBE_GWS_ERF_case_3"]

    def get_code(spin='unpolarized', shortrange=True):
        description = ["Implemented by YingXing Cheng.\n"]

        if spin == 'unpolarized':
            vars = [rho, sigma]
            zk_list = [_func(rho, sigma, mu, parameters) for _func in funcs]
            E_tot_list = [_zk * rho for _zk in zk_list]
            input = [var.name for var in vars]
            outputs = [['Ea'], ['Ea', 'd1Ea'], ['Ea', 'd1Ea', 'd2Ea']]
            prefixs = ['ESRX', 'D1ESRX', 'D2ESRX']
        else:
            vars = [rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb]
            E_tot_list = [_func(rho_a * 2, sigma_aa *4, mu, parameters) * rho_a
                          + _func(rho_b * 2, sigma_bb*4, mu, parameters) * rho_b
                          for _func in funcs]
            zk_list = [E_tot / (rho_a + rho_b) for E_tot in E_tot_list]
            input = [var.name for var in vars]
            outputs = [['E'], ['E', 'd1E'], ['E', 'd1E', 'd2E']]
            prefixs = ['ESRX_SPIN', 'D1ESRX_SPIN', 'D2ESRX_SPIN']

        nbv1 = len(vars)
        nbv2 = (nbv1 + 1) * nbv1 // 2
        diff_orders = [[1], [1, nbv1], [1, nbv1, nbv2]]

        d1E_all_cases, d2E_all_cases = get_derivative(E_tot_list, vars)

        for order in range(3):
            prefix = prefixs[order]
            output = outputs[order]
            diff_order = diff_orders[order]
            diff_idx = get_diff_idx_from_order(diff_order)
            func_names = [prefix +"_"+ _fn for _fn in fnames]

            for i, fn in enumerate(func_names):
                if order == 0:
                    kernel = [zk_list[i]]
                elif order == 1:
                    kernel = [zk_list[i]] + d1E_all_cases[i]
                elif order == 2:
                    kernel = [zk_list[i]] + d1E_all_cases[i] + d2E_all_cases[i]
                else:
                    raise RuntimeError('order should be less than 3!')
                dalprint.dalton_functional_printer(kernel, fn, input, output,
                                                   description=description, shortrange=shortrange,
                                                   diff_order=diff_order, diff_idx=diff_idx,
                                                   output_files=[out_file])


    get_code('unpolarized')
    get_code('polarized')
    out_file.close()

if "TPSS_ERFGWS_exchange" in write_list:
    out_file = open("../srfunctionals/TPSS_ERFGWS_exchange.F", "w+")
    # ######################################################################
    # TPSS exchange
    out_file.write("C SOURCES:\n")
    out_file.write(
        "C    Erich Goll, Matthias Ernst, Franzeska Moegle-Hofacker, and Hermann Stoll. \n")
    out_file.write("C    Development and assessment of a short-range meta-GGA functional.\n")
    out_file.write("C    The Journal of Chemical Physics, 130(23):234112, jun 2009.\n")
    out_file.write("\n")
    # ######################################################################
    funcs = [
        mu_func.TPSSx_mu_case_1,
        mu_func.TPSSx_mu_case_2_1,
        mu_func.TPSSx_mu_case_2_2,
        mu_func.TPSSx_mu_case_2_3,
        mu_func.TPSSx_mu_case_3
    ]

    fnames = ["TPSS_GWS_ERF_case_1",
              "TPSS_GWS_ERF_case_2_1",
              "TPSS_GWS_ERF_case_2_2",
              "TPSS_GWS_ERF_case_2_3",
              "TPSS_GWS_ERF_case_3"]

    def get_code(spin='unpolarized', shortrange=True):
        description = ["Implemented by YingXing Cheng.\n"]

        if spin == 'unpolarized':
            vars = [rho, sigma, tau, lapl]
            zk_list = [_func(rho, sigma, tau, lapl, mu, parameters) for _func in funcs]
            E_tot_list = [_zk * rho for _zk in zk_list]
            input = [var.name for var in vars]
            outputs = [['Ea'], ['Ea', 'd1Ea'], ['Ea', 'd1Ea', 'd2Ea']]
            prefixs = ['ESRX', 'D1ESRX', 'D2ESRX']
        else:
            vars = [rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb, tau_a, tau_b, lapl_a, lapl_b]
            E_tot_list = [
                _func(rho_a * 2, sigma_aa *4, tau*2, lapl*2, mu, parameters) * rho_a
                + _func(rho_b * 2, sigma_bb*4, tau*2, lapl*2, mu, parameters) * rho_b
                for _func in funcs
            ]
            zk_list = [E_tot / (rho_a + rho_b) for E_tot in E_tot_list]
            input = [var.name for var in vars]
            outputs = [['E'], ['E', 'd1E'], ['E', 'd1E', 'd2E']]
            prefixs = ['ESRX_SPIN', 'D1ESRX_SPIN', 'D2ESRX_SPIN']

        nbv1 = len(vars)
        nbv2 = (nbv1 + 1) * nbv1 // 2
        diff_orders = [[1], [1, nbv1], [1, nbv1, nbv2]]

        d1E_all_cases, d2E_all_cases = get_derivative(E_tot_list, vars)

        for order in range(3):
            prefix = prefixs[order]
            output = outputs[order]
            diff_order = diff_orders[order]
            diff_idx = get_diff_idx_from_order(diff_order)
            func_names = [prefix +"_"+ _fn for _fn in fnames]

            for i, fn in enumerate(func_names):
                if order == 0:
                    kernel = [zk_list[i]]
                elif order == 1:
                    kernel = [zk_list[i]] + d1E_all_cases[i]
                elif order == 2:
                    kernel = [zk_list[i]] + d1E_all_cases[i] + d2E_all_cases[i]
                else:
                    raise RuntimeError('order should be less than 3!')
                dalprint.dalton_functional_printer(kernel, fn, input, output,
                                                   description=description, shortrange=shortrange,
                                                   diff_order=diff_order, diff_idx=diff_idx,
                                                   output_files=[out_file])
    get_code('unpolarized')

    out_file.close()

if "PW92_ERF_correlation" in write_list:
    out_file = open("../srfunctionals/PW92_ERF_correlation.F", "w+")
    # ######################################################################
    # PW92 correlation, no-spin
    out_file.write("C SOURCES:\n")
    out_file.write(
        "C    Simone Paziani, Saverio Moroni, Paola Gori-Giorgi, and Giovanni B. Bachelet.\n")
    out_file.write(
        "C    Local-spin-densityfunctional for multideterminant density functional theory.\n")
    out_file.write("C    Physical Review B, 73(15), apr 2006.\n")
    out_file.write("\n")


    # ######################################################################

    def get_unpolar(shortrange=True):
        zk = mu_func.PW92c_mu(rho, 0, mu, parameters)
        E = zk * rho
        d1E_rho = E.diff(rho)
        d2E_rho2 = d1E_rho.diff(rho)

        input = ["rho"]
        for order in range(3):
            if order == 0:
                kernel = [zk]
                subroutine_name = 'ESRC_PW92_ERF'
                output = ['Ea']
                diff_order = [1]
            elif order == 1:
                kernel = [zk, d1E_rho]
                subroutine_name = 'D1ESRC_PW92_ERF'
                output = ['Ea', 'd1Ea']
                diff_order = [1, 1]
            elif order == 2:
                kernel = [zk, d1E_rho, d2E_rho2]
                subroutine_name = 'D2ESRC_PW92_ERF'
                output = ['Ea', 'd1Ea', 'd2Ea']
                diff_order = [1, 1, 1]
            else:
                raise RuntimeError('order larger than 2 has not been implemented!')
            diff_idx = get_diff_idx_from_order(diff_order)
            description = ["Implemented by YingXing Cheng.\n"]
            dalprint.dalton_functional_printer(kernel, subroutine_name, input, output,
                                               description=description, shortrange=shortrange,
                                               diff_order=diff_order, diff_idx=diff_idx,
                                               output_files=[out_file])


    # ######################################################################
    # PW92 correlation, spin
    # ######################################################################
    def get_polar(shortrange=True):
        rho_lis = [rho_a, rho_b]
        nb_rho = len(rho_lis)

        zk = mu_func.PW92c_mu(rho_a + rho_b, rho_a - rho_b, mu, parameters)
        E_tot = zk * (rho_a + rho_b)

        vrho = []
        for _rho in rho_lis:
            vrho.append(E_tot.diff(_rho))

        v2rho2 = []
        for i in range(nb_rho):
            for j in range(i, nb_rho):
                v2rho2.append(vrho[i].diff(rho_lis[j]))

        input = ["rho_a", "rho_b"]
        for order in range(3):
            if order == 0:
                kernel = [zk]
                subroutine_name = 'ESRC_SPIN_PW92_ERF'
                output = ['E']
                diff_order = [1]
            elif order == 1:
                kernel = [zk] + vrho
                subroutine_name = 'D1ESRC_SPIN_PW92_ERF'
                output = ['E', 'd1E']
                diff_order = [1, 2]
            elif order == 2:
                kernel = [zk] + vrho + v2rho2
                subroutine_name = 'D2ESRC_SPIN_PW92_ERF'
                output = ['E', 'd1E', 'd2E']
                diff_order = [1, 2, 3]
            else:
                raise RuntimeError('order larger than 2 has not been implemented!')
            diff_idx = get_diff_idx_from_order(diff_order)
            description = ["Implemented by YingXing Cheng.\n"]
            dalprint.dalton_functional_printer(kernel, subroutine_name, input, output,
                                               description=description, shortrange=shortrange,
                                               diff_order=diff_order, diff_idx=diff_idx,
                                               output_files=[out_file])


    get_unpolar()
    get_polar()

    # # # ######################################################################
    # # PW92 correlation, singlet-reference triplet response
    # # ######################################################################
    # E = mu_func.PW92c_mu(parameters) * rho_c
    # d1E_rhoc = E.diff(rho_c)
    # d1E_rhos = E.diff(rho_s)
    # d2E_rhoc2 = d1E_rhoc.diff(rho_c)
    # d2E_rhos2 = d1E_rhos.diff(rho_s)

    # Kernel = [E.subs({rho_s: 0}), d1E_rhoc.subs({rho_s: 0}), d2E_rhoc2.subs({rho_s: 0}),
    #           d2E_rhos2.subs({rho_s: 0})]
    # description = ["Implemented by E.R. Kjellgren.\n"]
    # diff_order = [1, 1, 2]
    # diff_idx = [[0], [1], [1, 3]]
    # dalprint.dalton_functional_printer(Kernel, "D2ESRC_PW92_ERF_singletref_triplet", ["rho_c"],
    #                                    ["E", "d1E", "d2E"], description=description,
    #                                    shortrange=True, diff_order=diff_order, diff_idx=diff_idx,
    #                                    output_files=[out_file])
    out_file.close()

if "VWN5_ERF_correlation" in write_list:
    out_file = open("../srfunctionals/VWN5_ERF_correlation.F", "w+")
    # ######################################################################
    # VWN5 correlation, no-spin
    out_file.write("C SOURCES:\n")
    out_file.write(
        "C    Simone Paziani, Saverio Moroni, Paola Gori-Giorgi, and Giovanni B. Bachelet.\n")
    out_file.write(
        "C    Local-spin-densityfunctional for multideterminant density functional theory.\n")
    out_file.write("C    Physical Review B, 73(15), apr 2006.\n")
    out_file.write("\n")


    # ######################################################################

    def get_unpolar():
        zk = mu_func.VWN5c_mu(rho, 0, mu, parameters)
        E = zk * rho
        d1E_rho = E.diff(rho)
        d2E_rho2 = d1E_rho.diff(rho)

        input = ["rho"]
        for order in range(3):
            if order == 0:
                kernel = [zk]
                subroutine_name = 'ESRC_VWN5_ERF'
                output = ['Ea']
                diff_order = [1]
            elif order == 1:
                kernel = [zk, d1E_rho]
                subroutine_name = 'D1ESRC_VWN5_ERF'
                output = ['Ea', 'd1Ea']
                diff_order = [1, 1]
            elif order == 2:
                kernel = [zk, d1E_rho, d2E_rho2]
                subroutine_name = 'D2ESRC_VWN5_ERF'
                output = ['Ea', 'd1Ea', 'd2Ea']
                diff_order = [1, 1, 1]
            else:
                raise RuntimeError('order larger than 2 has not been implemented!')
            diff_idx = get_diff_idx_from_order(diff_order)
            description = ["Implemented by YingXing Cheng.\n"]
            dalprint.dalton_functional_printer(kernel, subroutine_name, input, output,
                                               description=description, shortrange=True,
                                               diff_order=diff_order, diff_idx=diff_idx,
                                               output_files=[out_file])


    # ######################################################################
    # VWN5 correlation, spin
    # ######################################################################
    def get_polar():
        rho_lis = [rho_a, rho_b]
        nb_rho = len(rho_lis)

        zk = mu_func.VWN5c_mu(rho_a + rho_b, rho_a - rho_b, mu, parameters)
        E_tot = zk * (rho_a + rho_b)

        vrho = []
        for _rho in rho_lis:
            vrho.append(E_tot.diff(_rho))

        v2rho2 = []
        for i in range(nb_rho):
            for j in range(i, nb_rho):
                v2rho2.append(vrho[i].diff(rho_lis[j]))

        input = ["rho_a", "rho_b"]
        for order in range(3):
            if order == 0:
                kernel = [zk]
                subroutine_name = 'ESRC_SPIN_VWN5_ERF'
                output = ['E']
                diff_order = [1]
            elif order == 1:
                kernel = [zk] + vrho
                subroutine_name = 'D1ESRC_SPIN_VWN5_ERF'
                output = ['E', 'd1E']
                diff_order = [1, 2]
            elif order == 2:
                kernel = [zk] + vrho + v2rho2
                subroutine_name = 'D2ESRC_SPIN_VWN5_ERF'
                output = ['E', 'd1E', 'd2E']
                diff_order = [1, 2, 3]
            else:
                raise RuntimeError('order larger than 2 has not been implemented!')
            diff_idx = get_diff_idx_from_order(diff_order)
            description = ["Implemented by YingXing Cheng.\n"]
            dalprint.dalton_functional_printer(kernel, subroutine_name, input, output,
                                               description=description, shortrange=True,
                                               diff_order=diff_order, diff_idx=diff_idx,
                                               output_files=[out_file])


    get_unpolar()
    get_polar()

    # # ######################################################################
    # # VWN5 correlation, singlet-reference triplet response
    # # ######################################################################
    # E = mu_func.VWN5c_mu(parameters) * rho_c
    # d1E_rhoc = E.diff(rho_c)
    # d1E_rhos = E.diff(rho_s)
    # d2E_rhoc2 = d1E_rhoc.diff(rho_c)
    # d2E_rhos2 = d1E_rhos.diff(rho_s)

    # Kernel = [E.subs({rho_s: 0}), d1E_rhoc.subs({rho_s: 0}), d2E_rhoc2.subs({rho_s: 0}),
    #           d2E_rhos2.subs({rho_s: 0})]
    # description = ["Implemented by E.R. Kjellgren.\n"]
    # diff_order = [1, 1, 2]
    # diff_idx = [[0], [1], [1, 3]]
    # dalprint.dalton_functional_printer(Kernel, "D2ESRC_VWN5_ERF_singletref_triplet", ["rho_c"],
    #                                    ["E", "d1E", "d2E"], description=description,
    #                                    shortrange=True, diff_order=diff_order, diff_idx=diff_idx,
    #                                    output_files=[out_file])
    # out_file.close()

if "PBE_ERFGWS_correlation" in write_list:
    out_file = open("../srfunctionals/PBE_ERFGWS_correlation.F", "w+")
    # ######################################################################
    # PBE correlation, no-spin
    out_file.write("C SOURCES:\n")
    out_file.write("C    Erich Goll,  Hans-Joachim Werner, and Hermann Stoll.\n")
    out_file.write(
        "C    A short-range gradient-corrected densityfunctional in long-range coupled-cluster calculations for rare gas dimers.\n")
    out_file.write("C    Physical Chemistry Chemical Physics, 7(23):3917, 2005.\n")
    out_file.write("\n")
    out_file.write(
        "C    Erich Goll, Hans-Joachim Werner, Hermann Stoll, Thierry Leininger, Paola Gori-Giorgi, and Andreas Savin.\n")
    out_file.write(
        "C    A short-range gradient-corrected spin density functional in combination with long-range coupled-cluster methods:  Application to alkali-metal rare-gas dimers.\n")
    out_file.write("C    Chemical Physics,329(1-3):276-282, oct 2006.\n")
    out_file.write("\n")
    # ######################################################################
    zk = mu_func.PBEc_mu(rho, 0, sigma, mu, parameters)
    E_tot = zk * rho
    vrho = E_tot.diff(rho)
    vsigma = E_tot.diff(sigma)
    v2rho2 = vrho.diff(rho)
    v2rhosigma = vrho.diff(sigma)
    v2sigma2 = vsigma.diff(sigma)

    Kernel = [zk]
    description = ["Implemented by YingXing Cheng.\n"]
    diff_order = [1]
    diff_idx = get_diff_idx_from_order(diff_order)
    dalprint.dalton_functional_printer(Kernel, "ESRC_PBE_GWS_ERF", ["rho", "sigma"], ["Ea"],
                                       description=description, shortrange=True,
                                       diff_order=diff_order, diff_idx=diff_idx,
                                       output_files=[out_file])
    # # #
    Kernel = [zk, vrho, vsigma]
    description = ["Implemented by YingXing Cheng.\n"]
    diff_order = [1, 2]
    diff_idx = get_diff_idx_from_order(diff_order)
    dalprint.dalton_functional_printer(Kernel, "D1ESRC_PBE_GWS_ERF", ["rho", "sigma"],
                                       ["Ea", "d1Ea"], description=description, shortrange=True,
                                       diff_order=diff_order, diff_idx=diff_idx,
                                       output_files=[out_file])
    # # #
    Kernel = [zk, vrho, vsigma, v2rho2, v2rhosigma, v2sigma2]
    description = ["Implemented by YingXing Cheng.\n"]
    diff_order = [1, 2, 3]
    diff_idx = get_diff_idx_from_order(diff_order)
    dalprint.dalton_functional_printer(Kernel, "D2ESRC_PBE_GWS_ERF", ["rho", "sigma"],
                                       ["Ea", "d1Ea", "d2Ea"], description=description,
                                       shortrange=True, diff_order=diff_order, diff_idx=diff_idx,
                                       output_files=[out_file])

    # ######################################################################
    # PBE correlation, spin
    # ######################################################################
    rho_lis = [rho_a, rho_b]
    sigma_lis = [sigma_aa, sigma_ab, sigma_bb]
    nb_rho = len(rho_lis)
    nb_sigma = len(sigma_lis)

    zk = mu_func.PBEc_mu(rho_a + rho_b, rho_a - rho_b, sigma_aa + 2 * sigma_ab + sigma_bb, mu,
                         parameters)
    E_tot = zk * (rho_a + rho_b)

    vrho = []
    for _rho in rho_lis:
        vrho.append(E_tot.diff(_rho))

    vsigma = []
    for _sigma in sigma_lis:
        vsigma.append(E_tot.diff(_sigma))

    v2rho2 = []
    for i in range(nb_rho):
        for j in range(i, nb_rho):
            v2rho2.append(vrho[i].diff(rho_lis[j]))

    v2rhosigma = []
    for i in range(nb_rho):
        for j in range(nb_sigma):
            v2rhosigma.append(vrho[i].diff(sigma_lis[j]))

    v2sigma2 = []
    for i in range(nb_sigma):
        for j in range(i, nb_sigma):
            v2sigma2.append(vsigma[i].diff(sigma_lis[j]))

    Kernel = [zk]
    description = ["Implemented by YingXing Cheng.\n"]
    diff_order = [1]
    diff_idx = get_diff_idx_from_order(diff_order)
    dalprint.dalton_functional_printer(Kernel, "ESRC_SPIN_PBE_GWS_ERF",
                                       ["rho_a", "rho_b", "sigma_aa", "sigma_ab", "sigma_bb"],
                                       ["E"],
                                       description=description, shortrange=True,
                                       diff_order=diff_order, diff_idx=diff_idx,
                                       output_files=[out_file])
    # # #
    Kernel = [zk] + vrho + vsigma
    description = ["Implemented by YingXing Cheng.\n"]
    diff_order = [1, 5]
    diff_idx = get_diff_idx_from_order(diff_order)
    dalprint.dalton_functional_printer(Kernel, "D1ESRC_SPIN_PBE_GWS_ERF",
                                       ["rho_a", "rho_b", "sigma_aa", "sigma_ab", "sigma_bb"],
                                       ["E", "d1E"],
                                       description=description, shortrange=True,
                                       diff_order=diff_order, diff_idx=diff_idx,
                                       output_files=[out_file])
    # # #
    Kernel = [zk] + vrho + vsigma + v2rho2 + v2rhosigma + v2sigma2
    # Kernel = [E, d1E_rhoc, d1E_rhos, d1E_gammacc, d2E_rhoc2, d2E_rhocrhos, d2E_rhos2,
    #           d2E_rhocgammacc, d2E_rhosgammacc, d2E_gammacc2]
    description = ["Implemented by YingXing Cheng.\n"]
    diff_order = [1, 5, 15]
    diff_idx = get_diff_idx_from_order(diff_order)
    dalprint.dalton_functional_printer(Kernel, "D2ESRC_SPIN_PBE_GWS_ERF",
                                       ["rho_a", "rho_b", "sigma_aa", "sigma_ab", "sigma_bb"],
                                       ["E", "d1E", "d2E"],
                                       description=description, shortrange=True,
                                       diff_order=diff_order, diff_idx=diff_idx,
                                       output_files=[out_file])

    # # ######################################################################
    # # PBE correlation, singlet-reference triplet response
    # # ######################################################################
    # E = mu_func.PBEc_mu(parameters) * rho_c
    # d1E_rhoc = E.diff(rho_c)
    # d1E_rhos = E.diff(rho_s)
    # d1E_gammacc = E.diff(gamma_cc)
    # d2E_rhoc2 = d1E_rhoc.diff(rho_c)
    # d2E_rhos2 = d1E_rhos.diff(rho_s)
    # d2E_rhocgammacc = d1E_rhoc.diff(gamma_cc)
    # d2E_gammacc2 = d1E_gammacc.diff(gamma_cc)
    # Kernel = [E.subs({rho_s: 0}), d1E_rhoc.subs({rho_s: 0}), d1E_gammacc.subs({rho_s: 0}),
    #           d2E_rhoc2.subs({rho_s: 0}), d2E_rhos2.subs({rho_s: 0}),
    #           d2E_rhocgammacc.subs({rho_s: 0}), d2E_gammacc2.subs({rho_s: 0})]
    # description = ["Implemented by E.R. Kjellgren.\n"]
    # diff_order = [1, 2, 4]
    # diff_idx = [[0], [1, 3], [1, 3, 4, 6]]
    # dalprint.dalton_functional_printer(Kernel, "D2ESRC_PBE_GWS_ERF_singletref_triplet",
    #                                    ["rho_c", "gamma_cc"], ["E", "d1E", "d2E"],
    #                                    description=description, shortrange=True,
    #                                    diff_order=diff_order, diff_idx=diff_idx,
    #                                    output_files=[out_file])
    out_file.close()

if "TPSS_ERFGWS_correlation" in write_list:
    out_file = open("../srfunctionals/TPSS_ERFGWS_correlation.F", "w+")
    # ######################################################################
    # TPSS correlation, no-spin
    out_file.write("C SOURCES:\n")
    out_file.write(
        "C    Erich Goll, Matthias Ernst, Franzeska Moegle-Hofacker, and Hermann Stoll. \n")
    out_file.write("C    Development and assessment of a short-range meta-GGA functional.\n")
    out_file.write("C    The Journal of Chemical Physics, 130(23):234112, jun 2009.\n")
    out_file.write("\n")
    # ######################################################################
    description = ["Implemented by YingXing Cheng.\n"]

    funcs = [mu_func.TPSSc_mu_case_1,
             mu_func.TPSSc_mu_case_2,
             mu_func.TPSSc_mu_case_3,
             mu_func.TPSSc_mu_case_4]

    def get_unpolar_code():
        zk_list, E_tot_list = [], []
        for _func in funcs:
            zk = _func(rho, 0, sigma, 0, 0, tau, 0, mu, parameters)
            zk_list.append(zk)
            E_tot_list.append(zk * rho)

        d1E_all_cases, d2E_all_cases = get_derivative(E_tot_list, [rho, sigma, tau, lapl])
        cond_kernel = [func.PBEc(rho, 0, sigma, parameters),
                       spec_func.PBEc_alpha_replaced(rho, 0, sigma, 0, 0, parameters),
                       spec_func.PBEc_beta_replaced(rho, 0, sigma, 0, 0, parameters)]
        kernel0 = cond_kernel + zk_list

        diff_order = [1] * 4
        diff_idx = get_diff_idx_from_order(diff_order[:1]) * 4
        dalprint_spec.dalton_functional_printer(kernel0, "ESRC_TPSS_GWS_ERF",
                                                ["rho", "sigma", "tau", "lapl"],
                                                ["Ea"] * 4,
                                                description=description, shortrange=True,
                                                diff_order=diff_order, diff_idx=diff_idx,
                                                output_files=[out_file])
        # # #
        temp_list = []
        for i in range(4):
            temp_list.append(zk_list[i])
            temp_list.extend(d1E_all_cases[i])
        kernel1 = cond_kernel + temp_list

        diff_order = [1, len(d1E_all_cases[0])] * 4
        diff_idx = get_diff_idx_from_order(diff_order[:2]) * 4
        dalprint_spec.dalton_functional_printer(kernel1, "D1ESRC_TPSS_GWS_ERF",
                                                ["rho", "sigma", "tau", "lapl"],
                                                ["Ea", "d1Ea"] * 4,
                                                description=description, shortrange=True,
                                                diff_order=diff_order, diff_idx=diff_idx,
                                                output_files=[out_file])
        # # #
        temp2_list = []
        for i in range(4):
            temp2_list.append(zk_list[i])
            temp2_list.extend(d1E_all_cases[i])
            temp2_list.extend(d2E_all_cases[i])

        kernel2 = cond_kernel + temp2_list
        diff_order = [1, len(d1E_all_cases[0]), len(d2E_all_cases[0])] * 4
        diff_idx = get_diff_idx_from_order(diff_order[:3]) * 4
        dalprint_spec.dalton_functional_printer(kernel2, "D2ESRC_TPSS_GWS_ERF",
                                                ["rho", "sigma", "tau", "lapl"],
                                                ["Ea", "d1Ea", "d2Ea"] * 4,
                                                description=description, shortrange=True,
                                                diff_order=diff_order, diff_idx=diff_idx,
                                                output_files=[out_file])


    # ######################################################################
    # TPSS correlation, spin
    # ######################################################################
    def get_polar_code():
        zk_list, E_tot_list = [], []
        for _func in funcs:
            zk = _func(rho_a + rho_b, rho_a - rho_b, sigma_aa, sigma_ab, sigma_bb, tau_a, tau_b, mu,
                       parameters)
            zk_list.append(zk)
            E_tot_list.append(zk * (rho_a + rho_b))

        vars = [rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb, tau_a, tau_b, lapl_a, lapl_b]
        d1E_all_cases, d2E_all_cases = get_derivative(E_tot_list, vars)

        cond_kernel = [
            func.PBEc(rho_a + rho_b, rho_a - rho_b, sigma_aa + 2 * sigma_ab + sigma_bb, parameters),
            spec_func.PBEc_alpha_replaced(
                rho_a + rho_b, rho_a - rho_b, sigma_aa, sigma_ab, sigma_bb, parameters),
            spec_func.PBEc_beta_replaced(
                rho_a + rho_b, rho_a - rho_b, sigma_aa, sigma_ab, sigma_bb, parameters),
        ]
        kernel0 = cond_kernel + zk_list
        diff_order = [1] * 4
        diff_idx = get_diff_idx_from_order(diff_order[:1]) * 4
        dalprint_spec.dalton_functional_printer(kernel0, "ESRC_SPIN_TPSS_GWS_ERF",
                                                ["rho_a", "rho_b", "sigma_aa", "sigma_ab",
                                                 "sigma_bb",
                                                 "tau_a", "tau_b"],
                                                ["E"] * 4,
                                                description=description, shortrange=True,
                                                diff_order=diff_order, diff_idx=diff_idx,
                                                output_files=[out_file])
        # # #
        temp_list = []
        for i in range(4):
            temp_list.append(zk_list[i])
            temp_list.extend(d1E_all_cases[i])
        kernel1 = cond_kernel + temp_list
        diff_order = [1, len(d1E_all_cases[0])] * 4
        diff_idx = get_diff_idx_from_order(diff_order[:2]) * 4

        dalprint_spec.dalton_functional_printer(kernel1, "D1ESRC_SPIN_TPSS_GWS_ERF",
                                                ["rho_a", "rho_b", "sigma_aa", "sigma_ab",
                                                 "sigma_bb",
                                                 "tau_a", "tau_b"],
                                                ["E", "d1E"] * 4,
                                                description=description, shortrange=True,
                                                diff_order=diff_order, diff_idx=diff_idx,
                                                output_files=[out_file])
        # # #
        temp2_list = []
        for i in range(4):
            temp2_list.append(zk_list[i])
            temp2_list.extend(d1E_all_cases[i])
            temp2_list.extend(d2E_all_cases[i])

        kernel2 = cond_kernel + temp2_list
        diff_order = [1, len(d1E_all_cases[0]), len(d2E_all_cases[0])] * 4
        diff_idx = get_diff_idx_from_order(diff_order[:3]) * 4
        dalprint_spec.dalton_functional_printer(kernel2, "D2ESRC_SPIN_TPSS_GWS_ERF",
                                                ["rho_a", "rho_b", "sigma_aa", "sigma_ab",
                                                 "sigma_bb",
                                                 "tau_a", "tau_b"],
                                                ["E", "d1E", "d2E"] * 4,
                                                description=description, shortrange=True,
                                                diff_order=diff_order, diff_idx=diff_idx,
                                                output_files=[out_file])


    get_unpolar_code()

    # # ######################################################################
    # # TPSS correlation, singlet-reference triplet response
    # # ######################################################################
    # E_case_1 = mu_func.TPSSc_mu_case_1(parameters) * rho_c
    # d1E_rhoc_case_1 = E_case_1.diff(rho_c)
    # d1E_rhos_case_1 = E_case_1.diff(rho_s)
    # d1E_gammacc_case_1 = E_case_1.diff(gamma_cc)
    # d1E_gammass_case_1 = E_case_1.diff(gamma_ss)
    # d1E_gammacs_case_1 = E_case_1.diff(gamma_cs)
    # d1E_tauc_case_1 = E_case_1.diff(tau_c)
    # d2E_rhoc2_case_1 = d1E_rhoc_case_1.diff(rho_c)
    # d2E_rhos2_case_1 = d1E_rhos_case_1.diff(rho_s)
    # d2E_rhocgammacc_case_1 = d1E_rhoc_case_1.diff(gamma_cc)
    # d2E_gammacc2_case_1 = d1E_gammacc_case_1.diff(gamma_cc)
    # d2E_rhocgammass_case_1 = d1E_rhoc_case_1.diff(gamma_ss)
    # d2E_gammaccgammass_case_1 = d1E_gammacc_case_1.diff(gamma_ss)
    # d2E_gammass2_case_1 = d1E_gammass_case_1.diff(gamma_ss)
    # d2E_rhosgammacs_case_1 = d1E_rhos_case_1.diff(gamma_cs)
    # d2E_gammacs2_case_1 = d1E_gammacs_case_1.diff(gamma_cs)
    # d2E_rhoctauc_case_1 = d1E_rhoc_case_1.diff(tau_c)
    # d2E_gammacctauc_case_1 = d1E_gammacc_case_1.diff(tau_c)
    # d2E_gammasstauc_case_1 = d1E_gammass_case_1.diff(tau_c)
    # d2E_tauc2_case_1 = d1E_tauc_case_1.diff(tau_c)

    # E_case_2 = mu_func.TPSSc_mu_case_2(parameters) * rho_c
    # d1E_rhoc_case_2 = E_case_2.diff(rho_c)
    # d1E_rhos_case_2 = E_case_2.diff(rho_s)
    # d1E_gammacc_case_2 = E_case_2.diff(gamma_cc)
    # d1E_gammass_case_2 = E_case_2.diff(gamma_ss)
    # d1E_gammacs_case_2 = E_case_2.diff(gamma_cs)
    # d1E_tauc_case_2 = E_case_2.diff(tau_c)
    # d2E_rhoc2_case_2 = d1E_rhoc_case_2.diff(rho_c)
    # d2E_rhos2_case_2 = d1E_rhos_case_2.diff(rho_s)
    # d2E_rhocgammacc_case_2 = d1E_rhoc_case_2.diff(gamma_cc)
    # d2E_gammacc2_case_2 = d1E_gammacc_case_2.diff(gamma_cc)
    # d2E_rhocgammass_case_2 = d1E_rhoc_case_2.diff(gamma_ss)
    # d2E_gammaccgammass_case_2 = d1E_gammacc_case_2.diff(gamma_ss)
    # d2E_gammass2_case_2 = d1E_gammass_case_2.diff(gamma_ss)
    # d2E_rhosgammacs_case_2 = d1E_rhos_case_2.diff(gamma_cs)
    # d2E_gammacs2_case_2 = d1E_gammacs_case_2.diff(gamma_cs)
    # d2E_rhoctauc_case_2 = d1E_rhoc_case_2.diff(tau_c)
    # d2E_gammacctauc_case_2 = d1E_gammacc_case_2.diff(tau_c)
    # d2E_gammasstauc_case_2 = d1E_gammass_case_2.diff(tau_c)
    # d2E_tauc2_case_2 = d1E_tauc_case_2.diff(tau_c)

    # E_case_3 = mu_func.TPSSc_mu_case_3(parameters) * rho_c
    # d1E_rhoc_case_3 = E_case_3.diff(rho_c)
    # d1E_rhos_case_3 = E_case_3.diff(rho_s)
    # d1E_gammacc_case_3 = E_case_3.diff(gamma_cc)
    # d1E_gammass_case_3 = E_case_3.diff(gamma_ss)
    # d1E_gammacs_case_3 = E_case_3.diff(gamma_cs)
    # d1E_tauc_case_3 = E_case_3.diff(tau_c)
    # d2E_rhoc2_case_3 = d1E_rhoc_case_3.diff(rho_c)
    # d2E_rhos2_case_3 = d1E_rhos_case_3.diff(rho_s)
    # d2E_rhocgammacc_case_3 = d1E_rhoc_case_3.diff(gamma_cc)
    # d2E_gammacc2_case_3 = d1E_gammacc_case_3.diff(gamma_cc)
    # d2E_rhocgammass_case_3 = d1E_rhoc_case_3.diff(gamma_ss)
    # d2E_gammaccgammass_case_3 = d1E_gammacc_case_3.diff(gamma_ss)
    # d2E_gammass2_case_3 = d1E_gammass_case_3.diff(gamma_ss)
    # d2E_rhosgammacs_case_3 = d1E_rhos_case_3.diff(gamma_cs)
    # d2E_gammacs2_case_3 = d1E_gammacs_case_3.diff(gamma_cs)
    # d2E_rhoctauc_case_3 = d1E_rhoc_case_3.diff(tau_c)
    # d2E_gammacctauc_case_3 = d1E_gammacc_case_3.diff(tau_c)
    # d2E_gammasstauc_case_3 = d1E_gammass_case_3.diff(tau_c)
    # d2E_tauc2_case_3 = d1E_tauc_case_3.diff(tau_c)

    # E_case_4 = mu_func.TPSSc_mu_case_4(parameters) * rho_c
    # d1E_rhoc_case_4 = E_case_4.diff(rho_c)
    # d1E_rhos_case_4 = E_case_4.diff(rho_s)
    # d1E_gammacc_case_4 = E_case_4.diff(gamma_cc)
    # d1E_gammass_case_4 = E_case_4.diff(gamma_ss)
    # d1E_gammacs_case_4 = E_case_4.diff(gamma_cs)
    # d1E_tauc_case_4 = E_case_4.diff(tau_c)
    # d2E_rhoc2_case_4 = d1E_rhoc_case_4.diff(rho_c)
    # d2E_rhos2_case_4 = d1E_rhos_case_4.diff(rho_s)
    # d2E_rhocgammacc_case_4 = d1E_rhoc_case_4.diff(gamma_cc)
    # d2E_gammacc2_case_4 = d1E_gammacc_case_4.diff(gamma_cc)
    # d2E_rhocgammass_case_4 = d1E_rhoc_case_4.diff(gamma_ss)
    # d2E_gammaccgammass_case_4 = d1E_gammacc_case_4.diff(gamma_ss)
    # d2E_gammass2_case_4 = d1E_gammass_case_4.diff(gamma_ss)
    # d2E_rhosgammacs_case_4 = d1E_rhos_case_4.diff(gamma_cs)
    # d2E_gammacs2_case_4 = d1E_gammacs_case_4.diff(gamma_cs)
    # d2E_rhoctauc_case_4 = d1E_rhoc_case_4.diff(tau_c)
    # d2E_gammacctauc_case_4 = d1E_gammacc_case_4.diff(tau_c)
    # d2E_gammasstauc_case_4 = d1E_gammass_case_4.diff(tau_c)
    # d2E_tauc2_case_4 = d1E_tauc_case_4.diff(tau_c)

    # Kernel = [func.PBEc(parameters).subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           spec_func.PBEc_alpha_replaced(parameters).subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           spec_func.PBEc_beta_replaced(parameters).subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           E_case_1.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d1E_rhoc_case_1.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d1E_gammacc_case_1.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d1E_gammass_case_1.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d1E_tauc_case_1.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_rhoc2_case_1.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_rhos2_case_1.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_rhocgammacc_case_1.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_gammacc2_case_1.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_rhocgammass_case_1.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_gammaccgammass_case_1.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_gammass2_case_1.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_rhosgammacs_case_1.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_gammacs2_case_1.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_rhoctauc_case_1.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_gammacctauc_case_1.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_gammasstauc_case_1.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_tauc2_case_1.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           E_case_2.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d1E_rhoc_case_2.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d1E_gammacc_case_2.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d1E_gammass_case_2.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d1E_tauc_case_2.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_rhoc2_case_2.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_rhos2_case_2.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_rhocgammacc_case_2.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_gammacc2_case_2.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_rhocgammass_case_2.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_gammaccgammass_case_2.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_gammass2_case_2.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_rhosgammacs_case_2.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_gammacs2_case_2.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_rhoctauc_case_2.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_gammacctauc_case_2.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_gammasstauc_case_2.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_tauc2_case_2.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           E_case_3.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d1E_rhoc_case_3.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d1E_gammacc_case_3.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d1E_gammass_case_3.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d1E_tauc_case_3.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_rhoc2_case_3.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_rhos2_case_3.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_rhocgammacc_case_3.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_gammacc2_case_3.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_rhocgammass_case_3.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_gammaccgammass_case_3.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_gammass2_case_3.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_rhosgammacs_case_3.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_gammacs2_case_3.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_rhoctauc_case_3.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_gammacctauc_case_3.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_gammasstauc_case_3.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_tauc2_case_3.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           E_case_4.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d1E_rhoc_case_4.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d1E_gammacc_case_4.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d1E_gammass_case_4.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d1E_tauc_case_4.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_rhoc2_case_4.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_rhos2_case_4.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_rhocgammacc_case_4.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_gammacc2_case_4.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_rhocgammass_case_4.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_gammaccgammass_case_4.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_gammass2_case_4.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_rhosgammacs_case_4.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_gammacs2_case_4.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_rhoctauc_case_4.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_gammacctauc_case_4.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_gammasstauc_case_4.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0}),
    #           d2E_tauc2_case_4.subs({rho_s: 0, gamma_ss: 0, gamma_cs: 0})]
    # description = ["Implemented by E.R. Kjellgren.\n"]
    # diff_order = [1, 4, 13,
    #               1, 4, 13,
    #               1, 4, 13,
    #               1, 4, 13]
    # diff_idx = [[0], [1, 3, 4, 6], [1, 3, 4, 6, 7, 9, 10, 12, 15, 16, 17, 19, 21],
    #             [0], [1, 3, 4, 6], [1, 3, 4, 6, 7, 9, 10, 12, 15, 16, 17, 19, 21],
    #             [0], [1, 3, 4, 6], [1, 3, 4, 6, 7, 9, 10, 12, 15, 16, 17, 19, 21],
    #             [0], [1, 3, 4, 6], [1, 3, 4, 6, 7, 9, 10, 12, 15, 16, 17, 19, 21]]
    # dalprint_spec.dalton_functional_printer(Kernel, "D2ESRC_TPSS_GWS_ERF_singletref_triplet",
    #                                         ["rho_c", "gamma_cc", "tau_c"],
    #                                         ["E", "d1E", "d2E",
    #                                          "E", "d1E", "d2E",
    #                                          "E", "d1E", "d2E",
    #                                          "E", "d1E", "d2E"],
    #                                         description=description, shortrange=True,
    #                                         diff_order=diff_order, diff_idx=diff_idx,
    #                                         output_files=[out_file])
    out_file.close()

if "PBE_nomu_correlation" in write_list:
    out_file = open("../srfunctionals/PBE_nomu_correlation.F", "w+")
    # ######################################################################
    # PBE, mu=0.0, correlation
    out_file.write("C SOURCES:\n")
    out_file.write("C    John P. Perdew, Kieron Burke, and Matthias Ernzerhof.\n")
    out_file.write("C    Generalized gradient approximation made simple.\n")
    out_file.write("C    Physical Review Letters, 77(18):3865-3868, oct 1996.\n")
    out_file.write("\n")
    # ######################################################################
    zk = func.PBEc(rho, 0, sigma, parameters)
    E_tot = zk * rho
    vrho = E_tot.diff(rho)
    vsigma = E_tot.diff(sigma)
    v2rho2 = vrho.diff(rho)
    v2rhosigma = vrho.diff(sigma)
    v2sigma2 = vsigma.diff(sigma)

    Kernel = [zk]
    description = ["Implemented by YingXing Cheng.\n"]
    diff_order = [1]
    diff_idx = get_diff_idx_from_order(diff_order)
    dalprint.dalton_functional_printer(Kernel, "ESRC_PBE", ["rho", "sigma"], ["Ea"],
                                       description=description, shortrange=False,
                                       diff_order=diff_order, diff_idx=diff_idx,
                                       output_files=[out_file])

    # # #
    Kernel = [zk, vrho, vsigma]
    diff_order = [1, 2]
    diff_idx = get_diff_idx_from_order(diff_order)
    dalprint.dalton_functional_printer(Kernel, "D1ESRC_PBE", ["rho", "sigma"], ["Ea", "d1Ea"],
                                       description=description, shortrange=False,
                                       diff_order=diff_order, diff_idx=diff_idx,
                                       output_files=[out_file])

    # # #
    Kernel = [zk, vrho, vsigma, v2rho2, v2rhosigma, v2sigma2]
    diff_order = [1, 2, 3]
    diff_idx = get_diff_idx_from_order(diff_order)
    dalprint.dalton_functional_printer(Kernel, "D2ESRC_PBE", ["rho", "sigma"],
                                       ["Ea", "d1Ea", "d2Ea"], description=description,
                                       shortrange=False, diff_order=diff_order, diff_idx=diff_idx,
                                       output_files=[out_file])

    # ######################################################################
    # PBE correlatoin, spin
    # ######################################################################
    rho_lis = [rho_a, rho_b]
    sigma_lis = [sigma_aa, sigma_ab, sigma_bb]
    nb_rho = len(rho_lis)
    nb_sigma = len(sigma_lis)

    zk = func.PBEc(rho_a + rho_b, rho_a - rho_b, sigma_aa + 2 * sigma_ab + sigma_bb, parameters)
    E_tot = zk * (rho_a + rho_b)

    vrho = []
    for _rho in rho_lis:
        vrho.append(E_tot.diff(_rho))

    vsigma = []
    for _sigma in sigma_lis:
        vsigma.append(E_tot.diff(_sigma))

    v2rho2 = []
    for i in range(nb_rho):
        for j in range(i, nb_rho):
            v2rho2.append(vrho[i].diff(rho_lis[j]))

    v2rhosigma = []
    for i in range(nb_rho):
        for j in range(nb_sigma):
            v2rhosigma.append(vrho[i].diff(sigma_lis[j]))

    v2sigma2 = []
    for i in range(nb_sigma):
        for j in range(i, nb_sigma):
            v2sigma2.append(vsigma[i].diff(sigma_lis[j]))

    Kernel = [zk]
    diff_order = [1]
    diff_idx = get_diff_idx_from_order(diff_order)
    dalprint.dalton_functional_printer(Kernel, "ESRC_SPIN_PBE",
                                       ["rho_a", "rho_b", "sigma_aa", "sigma_ab", "sigma_bb"],
                                       ["E"], description=description, shortrange=False,
                                       diff_order=diff_order, diff_idx=diff_idx,
                                       output_files=[out_file])
    # # #
    Kernel = [zk] + vrho
    diff_order = [1, 5]
    diff_idx = get_diff_idx_from_order(diff_order)
    dalprint.dalton_functional_printer(Kernel, "D1ESRC_SPIN_PBE",
                                       ["rho_a", "rho_b", "sigma_aa", "sigma_ab", "sigma_bb"],
                                       ["E", "d1E"], description=description, shortrange=False,
                                       diff_order=diff_order, diff_idx=diff_idx,
                                       output_files=[out_file])
    # # #
    Kernel = [zk] + vrho + vsigma + v2rho2 + v2rhosigma + v2sigma2
    diff_order = [1, 5, 15]
    diff_idx = get_diff_idx_from_order(diff_order)
    dalprint.dalton_functional_printer(Kernel, "D2ESRC_SPIN_PBE",
                                       ["rho_a", "rho_b", "sigma_aa", "sigma_ab", "sigma_bb"],
                                       ["E", "d1E", "d2E"], description=description,
                                       shortrange=False, diff_order=diff_order, diff_idx=diff_idx,
                                       output_files=[out_file])

    # # ######################################################################
    # # PBE correlation, singlet-reference triplet response
    # # ######################################################################
    # E = func.PBEc(parameters) * rho_c
    # d1E_rhoc = E.diff(rho_c)
    # d1E_rhos = E.diff(rho_s)
    # d1E_gammacc = E.diff(gamma_cc)
    # d2E_rhoc2 = d1E_rhoc.diff(rho_c)
    # d2E_rhos2 = d1E_rhos.diff(rho_s)
    # d2E_rhocgammacc = d1E_rhoc.diff(gamma_cc)
    # d2E_gammacc2 = d1E_gammacc.diff(gamma_cc)
    # Kernel = [E.subs({rho_s: 0}), d1E_rhoc.subs({rho_s: 0}), d1E_gammacc.subs({rho_s: 0}),
    #           d2E_rhoc2.subs({rho_s: 0}), d2E_rhos2.subs({rho_s: 0}),
    #           d2E_rhocgammacc.subs({rho_s: 0}), d2E_gammacc2.subs({rho_s: 0})]
    # description = ["Implemented by E.R. Kjellgren.\n"]
    # diff_order = [1, 2, 4]
    # diff_idx = [[0], [1, 3], [1, 3, 4, 6]]
    # dalprint.dalton_functional_printer(Kernel, "D2ESRC_PBE_singletref_triplet",
    #                                    ["rho_c", "gamma_cc"], ["E", "d1E", "d2E"],
    #                                    description=description, shortrange=False,
    #                                    diff_order=diff_order, diff_idx=diff_idx,
    #                                    output_files=[out_file])
    out_file.close()

if "wPBE_exchange" in write_list:
    out_file = open("../srfunctionals/wPBE_exchange.F", "w+")
    # ######################################################################
    # wPBE exchange
    out_file.write("C SOURCES:\n")
    out_file.write(
        "C    Thomas  M.  Henderson,  Benjamin  G.  Janesko,  and  Gustavo  E.  Scuseria.\n")
    out_file.write(
        "C    Generalized  gradient approximation model exchange holes for range-separated hybrids.\n")
    out_file.write("C    The Journal of Chemical Physics,128(19):194105, may 2008.\n")
    out_file.write("\n")
    # ######################################################################
    # rho_c = 2*rho_a
    zk = mu_func.wPBEx(rho, sigma, mu, parameters)
    E = zk * rho
    d1E_rho = E.diff(rho)
    d1E_gamma = E.diff(sigma)
    d2E_rho2 = d1E_rho.diff(rho)
    d2E_rhogamma = d1E_rho.diff(sigma)
    d2E_gamma2 = d1E_gamma.diff(sigma)

    Kernel = [zk]
    description = ["Implemented by E.R. Kjellgren.\n"]
    diff_order = [1]
    diff_idx = get_diff_idx_from_order(diff_order)
    dalprint.dalton_functional_printer(Kernel, "ESRX_wPBE", ["rho", "sigma"], ["Ea"],
                                       description=description, shortrange=True,
                                       diff_order=diff_order, diff_idx=diff_idx,
                                       output_files=[out_file])
    # # #
    Kernel = [zk, d1E_rho, d1E_gamma]
    description = ["Implemented by E.R. Kjellgren.\n"]
    diff_order = [1, 2]
    diff_idx = get_diff_idx_from_order(diff_order)
    dalprint.dalton_functional_printer(Kernel, "D1ESRX_wPBE", ["rho", "sigma"], ["Ea", "d1Ea"],
                                       description=description, shortrange=True,
                                       diff_order=diff_order, diff_idx=diff_idx,
                                       output_files=[out_file])
    # # #
    Kernel = [zk, d1E_rho, d1E_gamma, d2E_rho2, d2E_rhogamma, d2E_gamma2]
    description = ["Implemented by E.R. Kjellgren.\n"]
    diff_order = [1, 2, 3]
    diff_idx = get_diff_idx_from_order(diff_order)
    dalprint.dalton_functional_printer(Kernel, "D2ESRX_wPBE", ["rho", "sigma"],
                                       ["Ea", "d1Ea", "d2Ea"],
                                       description=description, shortrange=True,
                                       diff_order=diff_order, diff_idx=diff_idx,
                                       output_files=[out_file])
    out_file.close()

if "VWN5_nomu_correlation" in write_list:
    out_file = open("../srfunctionals/VWN5_nomu_correlation.F", "w+")
    # ######################################################################
    # VWN5 correlation, no-spin
    out_file.write("C SOURCES:\n")
    out_file.write("C    S. H. Vosko, L. Wilk, and M. Nusair.\n")
    out_file.write(
        "C    Accurate spin-dependent electron liquid correlation energies for local spin density calculations: a critical analysis.\n")
    out_file.write("C    Canadian Journal of Physics, 58(8):1200-1211, aug 1980.\n")
    out_file.write("\n")
    # ######################################################################
    zk = func.VWN5c(rho, 0, parameters)
    E = zk * rho
    d1E_rho = E.diff(rho)
    d2E_rho2 = d1E_rho.diff(rho)

    Kernel = [zk]
    description = ["Implemented by YingXing Cheng.\n"]
    diff_order = [1]
    diff_idx = get_diff_idx_from_order(diff_order)
    dalprint.dalton_functional_printer(Kernel, "ESRC_VWN5", ["rho"], ["Ea"],
                                       description=description, shortrange=False,
                                       diff_order=diff_order, diff_idx=diff_idx,
                                       output_files=[out_file])
    # # #
    Kernel = [zk, d1E_rho]
    diff_order = [1, 1]
    diff_idx = get_diff_idx_from_order(diff_order)
    dalprint.dalton_functional_printer(Kernel, "D1ESRC_VWN5", ["rho"], ["Ea", "d1Ea"],
                                       description=description, shortrange=False,
                                       diff_order=diff_order, diff_idx=diff_idx,
                                       output_files=[out_file])
    # # #
    Kernel = [zk, d1E_rho, d2E_rho2]
    diff_order = [1, 1, 1]
    diff_idx = get_diff_idx_from_order(diff_order)
    dalprint.dalton_functional_printer(Kernel, "D2ESRC_VWN5", ["rho"], ["Ea", "d1Ea", "d2Ea"],
                                       description=description, shortrange=False,
                                       diff_order=diff_order, diff_idx=diff_idx,
                                       output_files=[out_file])

    # ######################################################################
    # VWN5 correlation, spin
    # ######################################################################
    rho_lis = [rho_a, rho_b]
    nb_rho = len(rho_lis)

    zk = func.VWN5c(rho_a + rho_b, rho_a - rho_b, parameters)
    E_tot = zk * (rho_a + rho_b)

    vrho = []
    for _rho in rho_lis:
        vrho.append(E_tot.diff(_rho))

    v2rho2 = []
    for i in range(nb_rho):
        for j in range(i, nb_rho):
            v2rho2.append(vrho[i].diff(rho_lis[j]))

    Kernel = [zk]
    diff_order = [1]
    diff_idx = get_diff_idx_from_order(diff_order)
    dalprint.dalton_functional_printer(Kernel, "ESRC_SPIN_VWN5", ["rho_a", "rho_b"], ["E"],
                                       description=description, shortrange=False,
                                       diff_order=diff_order, diff_idx=diff_idx,
                                       output_files=[out_file])
    # # #
    Kernel = [zk] + vrho
    diff_order = [1, 2]
    diff_idx = get_diff_idx_from_order(diff_order)
    dalprint.dalton_functional_printer(Kernel, "D1ESRC_SPIN_VWN5", ["rho_a", "rho_b"],
                                       ["E", "d1E"], description=description, shortrange=False,
                                       diff_order=diff_order, diff_idx=diff_idx,
                                       output_files=[out_file])
    # # #
    Kernel = [zk] + vrho + v2rho2
    diff_order = [1, 2, 3]
    diff_idx = get_diff_idx_from_order(diff_order)
    dalprint.dalton_functional_printer(Kernel, "D2ESRC_SPIN_VWN5", ["rho_a", "rho_b"],
                                       ["E", "d1E", "d2E"], description=description,
                                       shortrange=False, diff_order=diff_order, diff_idx=diff_idx,
                                       output_files=[out_file])

    # # ######################################################################
    # # VWN5 correlation, singlet-reference triplet response
    # # ######################################################################
    # E = func.VWN5c(parameters) * rho_c
    # d1E_rhoc = E.diff(rho_c)
    # d1E_rhos = E.diff(rho_s)
    # d2E_rhoc2 = d1E_rhoc.diff(rho_c)
    # d2E_rhos2 = d1E_rhos.diff(rho_s)

    # Kernel = [E.subs({rho_s: 0}), d1E_rhoc.subs({rho_s: 0}), d2E_rhoc2.subs({rho_s: 0}),
    #           d2E_rhos2.subs({rho_s: 0})]
    # description = ["Implemented by E.R. Kjellgren.\n"]
    # diff_order = [1, 1, 2]
    # diff_idx = [[0], [1], [1, 3]]
    # dalprint.dalton_functional_printer(Kernel, "D2ESRC_VWN5_singletref_triplet", ["rho_c"],
    #                                    ["E", "d1E", "d2E"], description=description,
    #                                    shortrange=False, diff_order=diff_order, diff_idx=diff_idx,
    #                                    output_files=[out_file])
    # out_file.close()

# ######################################################################
# ######################################################################
if 'TPSS_nomu_exchange' in write_list:
    out_file = open("../srfunctionals/TPSS_nomu_exchange.F", "w+")
    # ######################################################################
    # TPSS exchange
    out_file.write("C SOURCES:\n")
    out_file.write(
        "C    Erich Goll, Matthias Ernst, Franzeska Moegle-Hofacker, and Hermann Stoll. \n")
    out_file.write("C    Development and assessment of a short-range meta-GGA functional.\n")
    out_file.write("C    The Journal of Chemical Physics, 130(23):234112, jun 2009.\n")
    out_file.write("\n")
    # ######################################################################
    funcs = [
        func.TPSSx,
    ]

    fnames = ["TPSS"]

    def get_code(spin='unpolarized', shortrange=False):
        description = ["Implemented by YingXing Cheng.\n"]

        if spin == 'unpolarized':
            vars = [rho, sigma, tau, lapl]
            zk_list = [_func(rho, sigma, tau, lapl, parameters) for _func in funcs]
            E_tot_list = [_zk * rho for _zk in zk_list]
            input = [var.name for var in vars]
            outputs = [['Ea'], ['Ea', 'd1Ea'], ['Ea', 'd1Ea', 'd2Ea']]
            prefixs = ['ESRX', 'D1ESRX', 'D2ESRX']
        else:
            vars = [rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb, tau_a, tau_b, lapl_a, lapl_b]
            E_tot_list = [
                _func(rho_a * 2, sigma_aa * 4, tau * 2, lapl * 2, parameters) * rho_a
                + _func(rho_b * 2, sigma_bb * 4, tau * 2, lapl * 2, parameters) * rho_b
                for _func in funcs
            ]
            zk_list = [E_tot / (rho_a + rho_b) for E_tot in E_tot_list]
            input = [var.name for var in vars]
            outputs = [['E'], ['E', 'd1E'], ['E', 'd1E', 'd2E']]
            prefixs = ['ESRX_SPIN', 'D1ESRX_SPIN', 'D2ESRX_SPIN']

        nbv1 = len(vars)
        nbv2 = (nbv1 + 1) * nbv1 // 2
        diff_orders = [[1], [1, nbv1], [1, nbv1, nbv2]]

        d1E_all_cases, d2E_all_cases = get_derivative(E_tot_list, vars)

        for order in range(3):
            prefix = prefixs[order]
            output = outputs[order]
            diff_order = diff_orders[order]
            diff_idx = get_diff_idx_from_order(diff_order)
            func_names = [prefix + "_" + _fn for _fn in fnames]

            for i, fn in enumerate(func_names):
                if order == 0:
                    kernel = [zk_list[i]]
                elif order == 1:
                    kernel = [zk_list[i]] + d1E_all_cases[i]
                elif order == 2:
                    kernel = [zk_list[i]] + d1E_all_cases[i] + d2E_all_cases[i]
                else:
                    raise RuntimeError('order should be less than 3!')
                dalprint.dalton_functional_printer(kernel, fn, input, output,
                                                   description=description,
                                                   shortrange=shortrange,
                                                   diff_order=diff_order, diff_idx=diff_idx,
                                                   output_files=[out_file])


    get_code('unpolarized')

    out_file.close()
