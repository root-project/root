# run all vector operation
echo "run scalar " 
unset FASTM; unset OPT3; unset AUTOVEC; unset USEVC
make clean; make; ./vectorOperation


export FASTM=1
echo "run Autovec" 
make clean; make; ./vectorOperation


echo "Run Vc scalar" 
export USEVC=1; export VCSCALAR=1; 
make clean; make; ./vectorOperation



echo "Run Vc sse"
export NOAVX=1
unset VCSCALAR
make clean; make; ./vectorOperation


echo "run vc avx"
unset NOAVX
make clean; make; ./vectorOperation

