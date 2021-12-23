#! /bin/bash

host=`hostname`

if [[ "$NERSC_HOST" =~ .*perlmutter.* ]]; then
# CUDA 10.1 & cmake 3.18.0 together cause some problem with recognizing the `-pthread` flag.

    echo "$NERSC_HOST" matched perlmutter
    
    echo module load cray-python/3.9.4.2
    module load cray-python/3.9.4.2
    
fi
