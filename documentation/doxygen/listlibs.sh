#!/bin/sh

# Finding the system we are running

export DOXYGEN_LDD="ldd"

OS=`uname`

case "$OS" in
   "Linux") export DOXYGEN_LDD="ldd"
   ;;
   "Darwin")export DOXYGEN_LDD="otool -L"
   ;;
esac

# Transform collaboration diagram into list of libraries

echo '#!/bin/sh'  > listofclass.sh
echo '' >> listofclass.sh
grep -s "Collaboration diagram for"  $DOXYGEN_OUTPUT_DIRECTORY/html/class*.html | sed -e "s/.html:.*$//" | sed -e "s/^.*html\/class/\.\/makelibs.sh /"  >> listofclass.sh

chmod +x ./listofclass.sh
 ./listofclass.sh
