#!/bin/bash
# The Python tutorials appear in the Namespaces page. This script removes them from Namespaces.

HTMLPATH=$DOXYGEN_OUTPUT_DIRECTORY/html
if [ ! -d "$HTMLPATH" ]; then
   echo "Error: DOXYGEN_OUTPUT_DIRECTORY is not exported."
   exit 1
fi

sed -i -e /namespace$1.html/d $HTMLPATH/namespaces_dup.js
sed -i -e /namespace$1.html/d $HTMLPATH/namespaces.html
sed -i -e "/memberdecls/,+5d" $HTMLPATH/$1_8py.html

FILE=$HTMLPATH/namespace$1.html
if test -f "$FILE"; then
   rm $FILE
fi