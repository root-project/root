#!/bin/sh
# Modify all class overview pages of doxygen that have a "Collaboration diagram".
# This diagram is replaced by the list of libraries that this class uses.

# Finding the system we are running

listOfClasses=$(mktemp /tmp/listOfClasses.XXXXXX)

case "$1" in
   -j*)
      NJOB=${1#-j}
      ;;
esac

if [ ! -d "$DOXYGEN_OUTPUT_DIRECTORY" ]; then
   echo "Need to export DOXYGEN_OUTPUT_DIRECTORY"
   exit 1
fi

# Transform collaboration diagram into list of libraries
grep -sl "Collaboration diagram for" $DOXYGEN_OUTPUT_DIRECTORY/html/class*.html | sed -E "s/^.*html\/class([^[:space:]]+)\.html.*$/\1/" > ${listOfClasses}

if [ ! -s "${listOfClasses}" ]; then
   echo "No class found to modify"
   exit 0
fi

xargs -L 1 -P ${NJOB:-1} ./modifyClassWebpage.sh < ${listOfClasses}
