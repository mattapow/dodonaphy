#!/bin/bash
#
# compiles the stan program and applies the patch to the C++ to implement
# stochastic gradient descent
#
STAN_PREFIX=/home/koadman/software/cmdstan-2.21.0


$STAN_PREFIX/bin/stanc  --allow_undefined --include_paths=`pwd`/src/ --o=src/dodonaphy.hpp src/dodonaphy.stan && \
patch -p0 < src/dodonaphy_hpp.diff && \
g++ -std=c++1y -pthread -D_REENTRANT -Wno-sign-compare     -I $STAN_PREFIX/stan/lib/stan_math/lib/tbb_2019_U8/include -O3 -I $STAN_PREFIX/src -I $STAN_PREFIX/stan/src -I $STAN_PREFIX/lib/rapidjson_1.1.0/ -I $STAN_PREFIX/stan/lib/stan_math/ -I $STAN_PREFIX/stan/lib/stan_math/lib/eigen_3.3.3 -I $STAN_PREFIX/stan/lib/stan_math/lib/boost_1.69.0 -I $STAN_PREFIX/stan/lib/stan_math/lib/sundials_4.1.0/include  -I src/  -DBOOST_DISABLE_ASSERTS      -c  -x c++ -o src/dodonaphy.o src/dodonaphy.hpp && \
g++ -std=c++1y -pthread -D_REENTRANT -Wno-sign-compare     -I $STAN_PREFIX/stan/lib/stan_math/lib/tbb_2019_U8/include -O3 -I $STAN_PREFIX/src -I $STAN_PREFIX/stan/src -I $STAN_PREFIX/lib/rapidjson_1.1.0/ -I $STAN_PREFIX/stan/lib/stan_math/ -I $STAN_PREFIX/stan/lib/stan_math/lib/eigen_3.3.3 -I $STAN_PREFIX/stan/lib/stan_math/lib/boost_1.69.0 -I $STAN_PREFIX/stan/lib/stan_math/lib/sundials_4.1.0/include -I src/   -DBOOST_DISABLE_ASSERTS            -Wl,-L,"$STAN_PREFIX/stan/lib/stan_math/lib/tbb" -Wl,-rpath,"$STAN_PREFIX/stan/lib/stan_math/lib/tbb"  src/dodonaphy.o $STAN_PREFIX/src/cmdstan/main.o         $STAN_PREFIX/stan/lib/stan_math/lib/sundials_4.1.0/lib/libsundials_nvecserial.a $STAN_PREFIX/stan/lib/stan_math/lib/sundials_4.1.0/lib/libsundials_cvodes.a $STAN_PREFIX/stan/lib/stan_math/lib/sundials_4.1.0/lib/libsundials_idas.a $STAN_PREFIX/stan/lib/stan_math/lib/sundials_4.1.0/lib/libsundials_kinsol.a  $STAN_PREFIX/stan/lib/stan_math/lib/tbb/libtbb.so.2 -o bin/dodonaphy

