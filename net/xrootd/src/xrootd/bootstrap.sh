#!/bin/bash
#-------------------------------------------------------------------------------
# Author: Derek Feichtinger, 19 Oct 2005
# Rewritten by Lukasz Janyst <ljanyst@cern.ch>, 14 Feb 2011
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Find a program
#-------------------------------------------------------------------------------
function findProg()
{
  for prog in $@; do
    if test -x "`which $prog 2>/dev/null`"; then
      echo $prog
      break
    fi
  done
}

#-------------------------------------------------------------------------------
# Sanity checks
#-------------------------------------------------------------------------------
if test ! -e src/XrdVersion.hh.in; then
  echo "[!] Sanity check. Could not find src/XrdVersion.hh. You need to bootstrap from the xrootd main directory" 1>&2
  exit 1
fi

if test ! -e src/Makefile_include; then
  touch src/Makefile_include
fi

#-------------------------------------------------------------------------------
# Find autotools
#-------------------------------------------------------------------------------
LIBTOOLIZE=(libtoolize `findProg libtoolize glibtoolize` --copy --force)
ACLOCAL=(aclocal `findProg aclocal aclocal-1.10 aclocal-1.9`)
AUTOMAKE=(automake `findProg automake automake-1.10 automake-1.9` -acf)
AUTOCONF=(autoconf `findProg autoconf`)

CHAIN=(LIBTOOLIZE ACLOCAL AUTOMAKE AUTOCONF)

for PROG in ${CHAIN[*]}; do
  eval APP=\${$PROG[0]}
  eval EXEC=\${$PROG[1]}
  if test x$EXEC = x; then
    echo "[!] $APP not found. Please check your autotools configuration." 1>&2
    exit 1
  fi
done

#-------------------------------------------------------------------------------
# Run the bootstrap procedure
#-------------------------------------------------------------------------------
for PROG in ${CHAIN[*]}; do
  #-----------------------------------------------------------------------------
  # Evaluate the parameters
  #-----------------------------------------------------------------------------
  eval APP=\${$PROG[0]}
  eval EXEC=\${$PROG[1]}
  PARAMS=""
  # ARGH! no seq on mac! :(
  for i in 2 3 4 5 6 7 8 9 10; do
    eval PARAM=\${$PROG[$i]}
    if test x$PARAM = x; then break; fi
    PARAMS="$PARAMS $PARAM"
  done

  #-----------------------------------------------------------------------------
  # Execute and check the status
  #-----------------------------------------------------------------------------
  eval $EXEC $PARAMS
  if test ${?} -ne 0; then
    echo "[!] Unable to execute $APP" 1>&2
    echo "[!] $EXEC $PARAMS fails" 1>&2
    exit 2
  fi
done
