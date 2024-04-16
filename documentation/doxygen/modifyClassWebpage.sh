#!/bin/bash
# Replace the original collaboration diagram in the doxygen webpages by the diagram of used libraries.

HTMLPATH=$DOXYGEN_OUTPUT_DIRECTORY/html
DOXYGEN_LDD=${DOXYGEN_LDD:=ldd}
WORKFILE=$HTMLPATH/class${2}.html

if [ ! -d "$HTMLPATH" -o ! -f "$WORKFILE" ]; then
   echo "File $WORKFILE doesn't exist. Need DOXYGEN_OUTPUT_DIRECTORY exported."
   exit 1
fi

# Find the libraries for the class $2. $1 is the ROOT command
libname=$(${1} -l -b -q "${CMAKE_CURRENT_SOURCE_DIR}/libs.C+(\"$2\")" | grep 'mainlib=')

# The class was not found. Remove the collaboration graph
if [ -z "${libname}" ]; then
   echo "WARNING modifyClassWebpage.sh: libs.C could not get library of class $2 from ROOT interpreter. Removing its collaboration diagram."
   sed -i'.back' -e 's/^Collaboration diagram for.*$/<\/div>/g'  $WORKFILE
   sed -i'.back' '/__coll__graph.svg/I,+2 d'  $WORKFILE
   sed -i'.back' -e 's/<hr\/>The documentation for/<\/div><hr\/>The documentation for/g'  $WORKFILE
   rm $WORKFILE.back
   exit
fi

libname=${libname#mainlib=}
libname=${libname%.*}

# Picture name containing the "coll graph"
PICNAME="$HTMLPATH/${libname}__coll__graph.svg"

test -f "${PICNAME}" || { echo "Error: file $PICNAME not found."; exit 1; }

sed -i'.back' -e 's/Collaboration diagram for /Libraries for /g'  $WORKFILE
sed -i'.back' -e "s/class${2}__coll__graph.svg/${PICNAME##*/}/"  $WORKFILE

# Make sure the picture size in the html file the same as the svg
PICSIZE=`grep "svg width" $PICNAME | sed -e "s/<svg //"`
sed -i'.back' -e "s/\(^.*src\)\(.*__coll__graph.svg\"\)\( width.*\">\)\(.*$\)/<div class=\"center\"><img src\2 $PICSIZE><\/div>/" $WORKFILE
rm $WORKFILE.back
