#!/bin/bash

# A simple makeinstall script for xrootd. File discovery part.
#
# F.Furano (CERN IT/DM) , Jul 2009
#
# Parameters:
#  $1: path to the tree of the xrootd distro freshly compiled
#  $2: inc path
#  $3: bin path
#  $4: lib path
#  $5: etc path
#  $6: utils path



srcp=$1
incp=$2
binp=$3
libp=$4
etcp=$5
utilsp=$6

# Includes
awkcmd='{ print "'$srcp'/src"substr($1, 2)" '$incp'"substr($1, 2) }'
cd $srcp/src
if [ $? -ne 0 ]; then
 echo "Unable to find the header files"
 exit 1
fi
find . -type f -iname "*.hh" | awk "$awkcmd"
find . -type f -iname "*.icc" | awk "$awkcmd"

# Binaries
awkcmd='{ print "'$srcp'/bin"substr($1, 2)" '$binp'"substr($1, 2) }'
cd $srcp/bin
if [ $? -ne 0 ]; then
 echo "Unable to find the compiled binaries"
 exit 1
fi
find . -type f | awk "$awkcmd"

# Libs
awkcmd='{ print "'$srcp'/lib"substr($1, 2)" '$libp'"substr($1, 2) }'
cd $srcp/lib
if [ $? -ne 0 ]; then
 echo "Unable to find the compiled libraries"
 exit 1
fi
find . -type f | awk "$awkcmd"

# Etc
awkcmd='{ print "'$srcp'/etc"substr($1, 2)" '$etcp'"substr($1, 2) }'
cd $srcp/etc
if [ $? -ne 0 ]; then
 echo "Unable to find the etc directory"
 exit 1
fi
find . -type f | awk "$awkcmd"

# Utils
awkcmd='{ print "'$srcp'/utils"substr($1, 2)" '$utilsp'"substr($1, 2) }'
cd $srcp/utils
if [ $? -ne 0 ]; then
 echo "Unable to find the utils directory"
 exit 1
fi
find . -type f | awk "$awkcmd"