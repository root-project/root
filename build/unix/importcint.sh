#!/bin/sh

# Script to import new version of CINT in the ROOT CVS tree via a vendor branch.
# Called by main Makefile.
# If --dry-run we're doing a dry-run (no changes, just echo-ing).
# Optional argument can be cint7 or "cint" is assumed.
#
# Author: Axel Naumann 2006-12-06

# Exit nicely on request.
trap 'exit 0' 1 2 3 15

# This variable is utterly useless, and only meant for testing on a copy of the CVS repository.
# Should always be "root"
ROOTCVSROOT=root2

ECHO=echo
if test ! "x$1" = "x--dry-run"; then
   ECHO=
else
   shift
   echo '"--dry-run" specified - running in debug mode without an actual import.'
fi

TEMP="/tmp"
if test ! -d ${TEMP}; then
    echo "Cannot find TEMP directory ${TEMP}."
    exit 1
fi

USER=$LOGNAME
if test "x${USER}" = "x"; then
   USER=${USERNAME}
   if test "x${USER}" = "x"; then
      echo 'Neither $USER nor $USERNAME are set - but I need them.'
      exit 1
   fi
fi

CINT=$1
if test ! "x${CINT}" = "xcint7"; then
   CINT=cint
fi

if test "x${CINT}" = "xcint"; then
   REVFILTER='v5-[-[:digit:]]+'
   CINTVENDORBRANCH=CINT_VENDOR_BRANCH
else
   REVFILTER='v7-[-[:digit:]]+'
   CINTVENDORBRANCH=CINT7_VENDOR_BRANCH
fi

NEWTAG=$(cvs -z9 -q -d ':pserver:cvs@root.cern.ch:/user/cvs' rlog -l -h cint/inc/G__ci.h | \
	 egrep '^[[:space:]]'${REVFILTER}':' | \
	 head -n1 | \
	 sed -e 's,^[[:space:]]*v\(.*\):.*$,\1,' )

if test "x${NEWTAG}" = "x"; then
   echo "Cannot extract newest tag for ${CINT}"
   exit 1
fi

OLDVENDORTAG=$(cvs -z9 -q -d ':pserver:cvs@root.cern.ch:/user/cvs' rlog -l -h ${ROOTCVSROOT}/${CINT}/inc/G__ci.h | \
	       egrep '^[[:space:]]'${CINT}'.*:' | \
	       head -n1 | \
	       sed -e 's,^[[:space:]]*\(.*\):.*$,\1,' )

if test "x${OLDVENDORTAG}" = "x"; then
   echo "Cannot extract current vendor tag for root/${CINT}"
   exit 1
fi

if test "xcint${NEWTAG}" = "x${OLDVENDORTAG}"; then
   echo "The newest ${CINT} tag v${NEWTAG} and the previous vendor branch tag ${OLDVENDORTAG}"
   echo "indicate that CINT was not tagged, or that the newest tag is already imported"
   echo "into ROOT."
   exit 0
fi

OLDPWD=$(pwd)

echo "Importing ${CINT} revision ${NEWTAG} into vendor branch..."
cd ${TEMP} || exit 1
cvs -z9 -Q -d :pserver:cvs@root.cern.ch:/user/cvs checkout -r "v${NEWTAG}" cint || exit 1
find cint -type d -name 'CVS' -exec rm -rf {} \; -prune || exit 1

cd cint
${ECHO} cvs -z9 -q -d :ext:${USER}@root.cern.ch:/user/cvs import \
   -m "import v${NEWTAG}" -I! -Ireflex -ko \
   ${ROOTCVSROOT}/cint "${CINTVENDORBRANCH}" cint"${NEWTAG}" || exit 1

cd ..
rm -rf cint

echo "Updating ROOT/${CINT}..."

cvs -z9 -Q -d :ext:${USER}@root.cern.ch:/user/cvs checkout ${ROOTCVSROOT}/${CINT} || exit 1
cd ${ROOTCVSROOT}/${CINT} || exit 1

${ECHO} cvs update -j "${OLDVENDORTAG}" -j "cint${NEWTAG}" || exit 1

echo You will now have to run
echo '  cvs -z9 -q commit -m "apply changes from CINT vendor branch, importing cint'"${NEWTAG}"'"'
echo yourself, to upload the changes into ROOT.

exit 0
