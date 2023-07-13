#!/usr/bin/env bash

# Check command line arguments
if [ "$1" = "" ]; then
    cmd=`basename $0`
    echo "Usage: $cmd command [arguments]"
    exit 1
fi

# Print out the environment?
if [ "$WINE_WRAP_PRINTENV" != "on" ]; then
    export WINE_WRAP_PRINTENV=off
fi 

# Check O/S
if [ "$OS" = "Windows_NT" ]; then
    if [ "${SCRAM_ARCH:0:5}" != "win32" ]; then
        echo "ERROR! Invalid SCRAM_ARCH $SCRAM_ARCH on $OS for $0"
        exit 1
    fi
    if [ "$WINE_WRAP_DEBUG" != "" ]; then
        debug=1
    else
        debug=0
    fi
else
    debug=0
fi

# Make sure LOCALRT is defined
export LOCALRT=
export PATH=/afs/cern.ch/sw/lcg/app/spi/scram:"${PATH}"
eval `scram runtime -sh`
if [ "$LOCALRT" = "" ] ; then
    echo "ERROR! Cannot define LOCALRT - are you in a valid SCRAM directory?"
    exit 1
fi

# Keep the current SEAL options file if already defined
export SEAL_CONFIGURATION_FILE_OLD=${SEAL_CONFIGURATION_FILE}

# Load common functions
###echo "DEBUG: load common test functions..."
if [ -r $LOCALRT/config/test_functions.sh ] ; then
    . $LOCALRT/config/test_functions.sh
elif [ -r $LOCALRT/src/config/scram/test_functions.sh ] ; then
    . $LOCALRT/src/config/scram/test_functions.sh
else
    echo "ERROR! Cannot find common test functions"
    exit 1
fi
###echo "DEBUG: load common test functions... DONE"

## NB: Make sure $HOME/private/authentication.xml contains YOUR credentials!
check_authfile
set_preliminar_env

## By default ORACLE_HOME is unset
# keepORACLE_HOME=yes
prepare_env

# Fix the PATHs for specific systems (only if needed)
fix_win_paths

# Disable POOL trace files
#unset POOL_ORA_SQL_TRACE_ON 
#unset POOL_ORA_CERNTRACE_ON 

# Keep the current SEAL options file if already defined
if [ "$SEAL_CONFIGURATION_FILE_OLD" != "" ] ; then
    export SEAL_CONFIGURATION_FILE=$SEAL_CONFIGURATION_FILE_OLD
fi

#----------------------------------------------------------------------------
# Analyse SCRAM_ARCH and CMTCONFIG to determine if we should use WIN32
#----------------------------------------------------------------------------
echo "SCRAM_ARCH: $SCRAM_ARCH"
echo "CMTCONFIG:  $CMTCONFIG"
useWin32=0
if [ "${SCRAM_ARCH:0:5}" = "win32" ]; then
    useWin32=1
elif [ "$SCRAM_ARCH" = "" ] && [ "${CMTCONFIG:0:5}" = "win32" ]; then
    useWin32=1
fi

#----------------------------------------------------------------------------
# Unix platform
#----------------------------------------------------------------------------
if [ "$useWin32" = "0" ]; then

    ###echo "__UNIX Current environment variables: "; env
    echo "__UNIX Execute command: " $@
    $@
    
#----------------------------------------------------------------------------
# Win32 platform
#----------------------------------------------------------------------------
else

    # Debugging information
    ###echo "__WINE Current environment variables: "; env
    if [ "$OS" = "Windows_NT" ]; then
        echo "__WINE.NT Execute command: " $@
    else
        echo "__WINE.WINE Execute command: " $@
    fi

    # Execute the command under Wine
    if [ "$debug" = "0" ]; then
        cmd_wrapper $@
        status=${?}
    else
        debug $@
        status=${?}
    fi
    ###echo status=$status
    exit $status
    
#----------------------------------------------------------------------------
# End check on platform
#----------------------------------------------------------------------------
fi

