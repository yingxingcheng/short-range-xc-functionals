#!/bin/bash

# f2py PBE_ERFGWS_correlation.F -m pbe_erfgws_c -h pbe_erfgws_c.pyf
# f2py -c pbe_erfgws_c.pyf PBE_ERFGWS_correlation.F

# fname=PBE_ERFGWS_correlation
function main(){
    fname=$1
    suffix=F
    gfortran -c ${fname}.${suffix}
    # gfortran -c PBE_ERFGWS_exchange.F
    # f2py srdftfun.F -m srdftfun -h srdftfun.pyf
    # f2py -c -I/Users/yxcheng/softwares/src/dalton/DALTON/include srdftfun.pyf  PBE_ERFGWS_correlation.F PBE_ERFGWS_exchange.F srdftfun.F

    f2py ${fname}.F -m ${fname} -h ${fname}.pyf
    f2py -c  ${fname}.pyf  ${fname}.F
}

# main LDA_ERF_exchange
# main PBE_ERFGWS_correlation
main $1
