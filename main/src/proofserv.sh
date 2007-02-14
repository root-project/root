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
# If requested, initialize the environment.
#

if [ -n "$PROOF_INITCMD" ]; then
   # echo "init with $PROOF_INITCMD" >> /tmp/proofserv.log
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

# echo "$WRAPPER $ROOTSYS/bin/proofserv.exe $@" >> /tmp/proofserv.log

exec $WRAPPER $ROOTSYS/bin/proofserv.exe "$@"
