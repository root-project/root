#! /bin/csh

cd ..
mkdir -p include; 
cd include;
if ( ! -l TMVA ) then
    ln -s ../src TMVA
endif
cd -

# set symbolic links to data file and to rootmaps
cd test
if ( ! -l tmva_example.root ) then
    ln -s data/toy_sigbkg.root tmva_example.root
endif
if ( ! -l tmva_reg_example.root ) then
    ln -s data/regression_parabola_noweights.root tmva_reg_example.root
endif
if ( ! -l libTMVA.rootmap ) then
    ln -s ../lib/libTMVA.rootmap
    ln -s ../lib/libTMVA.rootmap .rootmap
endif
cd -

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
        setenv DYLD_LIBRARY_PATH $PWD/lib:${DYLD_LIBRARY_PATH}
    else 
        setenv DYLD_LIBRARY_PATH $PWD/lib:`root-config --libdir`
    endif

else if ( `root-config --platform` == "solaris" ) then

    if ($?LD_LIBRARY_PATH) then
        setenv LD_LIBRARY_PATH $PWD/lib:${LD_LIBRARY_PATH}
    else 
        setenv LD_LIBRARY_PATH $PWD/lib:`root-config --libdir`
    endif

else

    # The ROOTSYS/lib may be set in a LD_LIBRARY_PATH or using ld.so

    grep -q `echo $ROOTSYS/lib /etc/ld.so.cache`
    set root_in_ld = $status
    if ($?LD_LIBRARY_PATH) then
        setenv LD_LIBRARY_PATH $PWD/lib:${LD_LIBRARY_PATH}
    else 
        if ( ${root_in_ld} == 1 ) then
            setenv LD_LIBRARY_PATH $PWD/lib:`root-config --libdir`
        else
            setenv LD_LIBRARY_PATH $PWD/lib
        endif
    endif

endif

# prepare for PyROOT

if ($?PYTHONPATH) then
    setenv PYTHONPATH ${PWD}/lib:`root-config --libdir`:${PYTHONPATH}
else
    setenv PYTHONPATH ${PWD}/lib:`root-config --libdir`/lib
endif

cd test

