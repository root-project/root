#!/bin/bash

# A simple makeinstall script for xrootd
#
# F.Furano (CERN IT/DM) , Jul 2009
#
# Parameters:
#  $1: path to the tree of the xrootd distro freshly compiled
#  $2: path to where to install the stuff (e.g. /opt/xrootd)
#
#  Optional envvars:
#  $INCPATH: path where to install the headers (e.g. /usr/local/include/xrootd)
#  $BINPATH: path where to install the executables (e.g. /usr/local/bin)
#  $LIBPATH: path where to install the libraries (e.g. /usr/local/lib)
#  $ETCPATH: path where to install the etc stuff
#  $UTILSPATH: path where to install the utils stuff
cd $1
if [ $? -ne 0 ]; then
 echo "Invalid path for the source distribution"
 exit 1
fi

srcp="`pwd`"
echo "Distribution path: $srcp"

mkdir -p $2
if [ $? -ne 0 ]; then
 echo "Invalid path for the source distribution"
 exit 1
fi

cd $2
if [ $? -ne 0 ]; then
 echo "Invalid destination path"
 exit 1
fi

destp="`pwd`"
echo "Destination path: $destp"

rm -f $srcp/installed_files.tmp
touch $srcp/installed_files.tmp

# Prepare the final destination paths
incp=$INCPATH
binp=$BINPATH
libp=$LIBPATH
etcp=$ETCPATH
utilsp=$UTILSPATH

if [ "x$incp" = "x" ]; then
 incp=$destp/include/xrootd
fi

if [ "x$binp" = "x" ]; then
 binp=$destp/bin
fi

if [ "x$libp" = "x" ]; then
 libp=$destp/lib
fi

if [ "x$etcp" = "x" ]; then
 etcp=$destp/etc/xrootd
fi

if [ "x$utilsp" = "x" ]; then
 utilsp=$binp/xrootdutils
fi


mkdir -p $incp
if [ $? -ne 0 ]; then
 echo "Invalid includes path"
 exit 1
fi

mkdir -p $binp
if [ $? -ne 0 ]; then
 echo "Invalid binaries path"
 exit 1
fi

mkdir -p $libp
if [ $? -ne 0 ]; then
 echo "Invalid libs path"
 exit 1
fi

mkdir -p $etcp
if [ $? -ne 0 ]; then
 echo "Invalid etc path"
 exit 1
fi

mkdir -p $utilsp
if [ $? -ne 0 ]; then
 echo "Invalid utils path"
 exit 1
fi

echo
echo "------- Final destination paths:"
echo " includes   :$incp"
echo " binaries   :$binp"
echo " libraries  :$libp"
echo " etc        :$etcp"
echo " utils      :$utilsp"
echo

cd $srcp
rm -f ./makeinstall_filelist.log
./makeinstall_listfiles.sh $srcp $incp $binp $libp $etcp $utilsp > makeinstall_filelist.log
if [ $? -ne 0 ]; then
 echo "File list creation failed"
 exit 1
fi

cat makeinstall_filelist.log | awk '{print $2}' | xargs -n 1 dirname | xargs -n 1 mkdir -p
cat makeinstall_filelist.log | xargs -n 2 cp
