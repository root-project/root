#!/bin/csh -f

setenv HERE $PWD


if ( $#argv != 1 )  then
  cd .. ; setenv TMVASYS $PWD; cd $HERE 
  set TMVATESTDIR=1
  echo $TMVASYS
  echo $PWD
  if (( "$TMVASYS/test" != "$PWD" )) then
      echo
      echo "  please give the directory of your TMVA installation you want to use as argument to  "
      echo "  source setup.csh <the TMVA installation directory>"
      echo
      exit
  endif
else
  setenv TMVASYS $argv[1]
  set TMVATESTDIR=0
  echo 
  echo "  you have specified to use TMVA installed in:" $argv[1]
endif

# check if the TMVA directory specified REALLY contains the TMVA libraries, otherwise it
# might default to the ROOT version causing unnecessary surprises

if (( ! -f $TMVASYS/lib/libTMVA.so )) then
    echo 
    echo "  please give a PROPER directory of your TMVA installation as argument to  "
    echo "  source setup.csh <the TMVA installation directory> "
    echo 
    echo "  currently I look at $TMVASYS/lib/libTMVA.so  which doesn't exist "
    echo
    exit
endif


echo "use TMVA version installed in " $TMVASYS


# set symbolic links to data file and to rootmaps
#cd test;
if (( (!  -l tmva_example.root)  && ($TMVATESTDIR == 1 ) )) then 
    ln -s data/toy_sigbkg.root tmva_example.root
 endif
if (( (! -l tmva_reg_example.root) && ($TMVATESTDIR == 1) )) then 
    ln -s data/regression_parabola_noweights.root tmva_reg_example.root
 endif
if (( ! -l libTMVA.rootmap )) then 
    ln -s $TMVASYS/lib/libTMVA.rootmap
endif
if (( ! -l .rootmap )) then 
    ln -s $TMVASYS/lib/libTMVA.rootmap .rootmap
endif
if (( ! -f TMVAlogon.C )) then 
    cp $TMVASYS/test/TMVAlogon.C . 
endif
if (( ! -f TMVAGui.C )) then 
    cp $TMVASYS/test/TMVAGui.C . 
endif
if (( ! -f TMVARegGui.C )) then 
    cp $TMVASYS/test/TMVARegGui.C . 
endif
if (( ! -f tmvaglob.C )) then 
    cp $TMVASYS/test/tmvaglob.C . 
endif
if (( ! -f .rootrc )) then 
    cp $TMVASYS/test/.rootrc . 
endif


# Check Root environment setup
# It's checked in such a fancy way, because if you install ROOT using
# Fink on Mac OS X, thet will not setup the ROOTSYS environment variable.
# (The layout of ROOT is a bit different in that case...)
if ((!($?ROOTSYS)) && (! -e `which root-config`)) then
    echo "ERROR: Root environment not yet defined."
    echo "Please do so by setting the ROOTSYS environment variable to the"
    echo "directory of your root installation or to ROOT on CERN-afs (see http://root.cern.ch)! "
    exit 1
endif

# On MacOS X $DYLD_LIBRARY_PATH has to be modified, so:
if ( `root-config --platform` == "macosx" ) then

    if ($?DYLD_LIBRARY_PATH) then
        setenv DYLD_LIBRARY_PATH $TMVASYS/lib:${DYLD_LIBRARY_PATH}
    else 
        setenv DYLD_LIBRARY_PATH $TMVASYS/lib:`root-config --libdir`
    endif

else if ( `root-config --platform` == "solaris" ) then

    if ($?LD_LIBRARY_PATH) then
        setenv LD_LIBRARY_PATH $TMVASYS/lib:${LD_LIBRARY_PATH}
    else 
        setenv LD_LIBRARY_PATH $TMVASYS/lib:`root-config --libdir`
    endif

else

    # The ROOTSYS/lib may be set in a LD_LIBRARY_PATH or using ld.so

    grep -q `echo $ROOTSYS/lib /etc/ld.so.cache`
    set root_in_ld=$status
    if ($?LD_LIBRARY_PATH) then
        setenv LD_LIBRARY_PATH $TMVASYS/lib:${LD_LIBRARY_PATH}
    else 
        if ( ${root_in_ld} == 1 ) then
            setenv LD_LIBRARY_PATH $TMVASYS/lib:`root-config --libdir`
        else
            setenv LD_LIBRARY_PATH $TMVASYS/lib
        endif
    endif

endif

# prepare for PyROOT

if ($?PYTHONPATH) then
    setenv PYTHONPATH ${TMVASYS}/lib:`root-config --libdir`:${PYTHONPATH}
else
    setenv PYTHONPATH ${TMVASYS}/lib:`root-config --libdir`/lib
endif


cd $HERE
