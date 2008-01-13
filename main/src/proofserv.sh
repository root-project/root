#!/bin/sh

#
# $Id: proofserv.sh,v 1.1 2006/11/20 15:56:35 rdm Exp $
#
# The proofserv wrapper script can be used to initialize the
# environment for proofserv as needed. It could be extended
# to select a specific ROOT version. It also allows debug
# tools like valgrind etc to be used.
#
# This example script should be sufficient for most installations
# but can be modified as needed.
#

#
# Setup notification
#
if [ -n "$ROOTPROOFLOGFILE" ]; then
# Use the standard log file
   LOGFILE="$ROOTPROOFLOGFILE"
else
# Use a temp file
   if [ -n "$TMPDIR" ]; then
      LOGFILE="$TMPDIR/proofserv.log"
   else
      LOGFILE="/tmp/proofserv.log"
   fi
fi

#
# If requested, initialize the environment.
# The PROOF_INITCMD variable contains the command to be executed to initialize
# the environment.
# For a simple variable setting just use 'echo export VAR=value', e.g.
#
# root [] TProof::AddEnvVar("PROOF_INITCMD",
#               "echo export LD_LIBRARY_PATH=/some/new/libpath:$LDLIBRARY_PATH")
#
# If the setup is defined by a script, e.g. /some/path/setup-env.sh, then the
# script should be sourced:
#
# root [] TProof::AddEnvVar("PROOF_INITCMD","source /some/path/setup-env.sh")
#
# If the script outputs the command to be executed, e.g. /some/path/getscram.sh,
# then just put the script path:
#
# root [] TProof::AddEnvVar("PROOF_INITCMD","/some/path/getscram.sh")
#

if [ -n "$PROOF_INITCMD" ]; then
   NOW=`date +%H:%M:%S`
   echo "$NOW: $1: initializing environment with: $PROOF_INITCMD" >> "$LOGFILE"
   eval `$PROOF_INITCMD`
fi

#
# Run master, workers or all with a debug command.
# E.g in the client do:
#   root [] TProof::AddEnvVar("PROOF_WRAPPERCMD","valgrind --log-file=/tmp/vg") 
#
if [ -n "$PROOF_WRAPPERCMD" ]; then
   WRAPPER="$PROOF_WRAPPERCMD "
else
   if [ -n "$PROOF_MASTER_WRAPPERCMD" -a "$1" = "proofserv" ]; then
      WRAPPER="$PROOF_MASTER_WRAPPERCMD "
   fi
   if [ -n "$PROOF_SLAVE_WRAPPERCMD" -a "$1" = "proofslave" ]; then
      WRAPPER="$PROOF_SLAVE_WRAPPERCMD "
   fi
fi

if [ "x$WRAPPER" != "x" ]; then
   NOW=`date +%H:%M:%S`
   echo "$NOW: $1: executing $WRAPPER $ROOTSYS/bin/proofserv.exe $@" >> $LOGFILE
fi

exec $WRAPPER $ROOTSYS/bin/proofserv.exe "$@"
