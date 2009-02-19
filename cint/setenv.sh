export PATH=$PWD/bin:$PATH
export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
if uname | grep CYGWIN > /dev/null; then
	export CINTSYSDIR=`cygpath -m $PWD`
else
	export CINTSYSDIR=$PWD
fi
