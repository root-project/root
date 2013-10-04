#!/bin/sh

export HERE=$PWD

if [ $# -ne 1 ] ; then
  cd .. ; export TMVASYS=$PWD; cd $HERE 
  TMVATESTDIR=1
  if [[ "$TMVASYS/test" != "$PWD" ]]; then
      echo
      echo "!!! please give the directory of your TMVA installation you want to use as argument to  "
      echo "!!! source setup.sh <the TMVA installation directory>"
      echo
      return
  fi
else
  export TMVASYS=$1
  TMVATESTDIR=0
  echo 
  echo "  you have specified to use TMVA installed in:" $argv[1]
fi

# check if the TMVA directory specified REALLY contains the TMVA libraries, otherwise it
# might default to the ROOT version causing unnecessary surprises

if [[ ! -f $TMVASYS/lib/libTMVA.so ]]; then
    echo
    echo "!!!! please give a PROPER directory of your TMVA installation as argument to  "
    echo "!!!! source setup.sh <the TMVA installation directory> "
    echo 
    echo "!!!! currently I look at $TMVASYS/lib/libTMVA.so  that doesn't exist "
    echo
    return
fi


echo "use TMVA version installed in " $TMVASYS


# set symbolic links to data file and to rootmaps
#cd test;
if [[ !  -h tmva_example.root  && $TMVATESTDIR -eq 1 ]]; then ln -s data/toy_sigbkg.root tmva_example.root; fi
if [[ ! -h tmva_reg_example.root && $TMVATESTDIR -eq 1 ]]; then ln -s data/regression_parabola_noweights.root tmva_reg_example.root; fi
ln -sf $TMVASYS/lib/libTMVA.rootmap;
ln -sf $TMVASYS/lib/libTMVA.rootmap .rootmap; 
if [[ ! -f TMVAlogon.C ]]; then cp $TMVASYS/test/TMVAlogon.C . ; fi
if [[ ! -f TMVAGui.C ]]; then cp $TMVASYS/test/TMVAGui.C . ; fi
if [[ ! -f TMVARegGui.C ]]; then cp $TMVASYS/test/TMVARegGui.C . ; fi
if [[ ! -f tmvaglob.C ]]; then cp $TMVASYS/test/tmvaglob.C . ; fi
if [[ ! -f .rootrc ]]; then cp $TMVASYS/test/.rootrc . ; fi


# Check Root environment setup
# It's checked in such a fancy way, because if you install ROOT using
# Fink on Mac OS X, thet will not setup the ROOTSYS environment variable.
# (The layout of ROOT is a bit different in that case...)
#if test ! "$ROOTSYS" -a ! -e `which root-config`; then
#if [[ ! $ROOTSYS && ! -e `which root-config` ]]; then  ### not shell conform (shell on lxplus is bash, don't test there)
if [ ! $ROOTSYS ]; then
    echo "ERROR: Root environment not yet defined."
    echo "Please do so by setting the ROOTSYS environment variable to the"
    echo "directory of your root installation or to ROOT on CERN-afs (see http://root.cern.ch)! "
    return 1
fi



# On MacOS X $DYLD_LIBRARY_PATH has to be modified, so:
if [[ `root-config --platform` == "macosx" ]]; then

    if [ ! $DYLD_LIBRARY_PATH ]; then
        export DYLD_LIBRARY_PATH=$TMVASYS/lib:`root-config --libdir`
    else
        export DYLD_LIBRARY_PATH=$TMVASYS/lib:${DYLD_LIBRARY_PATH}
    fi

elif [[ `root-config --platform` == "solaris" ]]; then

    if [ ! $LD_LIBRARY_PATH ]; then
        export LD_LIBRARY_PATH=$TMVASYS/lib:`root-config --libdir`
    else
        export LD_LIBRARY_PATH=$TMVASYS/lib:${LD_LIBRARY_PATH}
    fi

else
    grep -q `echo $ROOTSYS/lib` /etc/ld.so.cache
    root_in_ld=$?
    if [ ! $LD_LIBRARY_PATH ]; then
        if [ $root_in_ld -ne 0 ]; then
            echo "Warning: so far you haven't setup your ROOT enviroment properly (no LD_LIBRARY_PATH): TMVA will not work"
        fi
    fi
    export LD_LIBRARY_PATH=$TMVASYS/lib:${LD_LIBRARY_PATH}

fi

# prepare for PyROOT
export PYTHONPATH=$TMVASYS/lib:`root-config --libdir`:$PYTHONPATH

cd $HERE
