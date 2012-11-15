#!/usr/bin/env bash

# Check version of interpreter/llvm directory and if it is changed force
# a make in the LLVM tree. It creates the file
#    interpreter/llvm/obj/llvmrev.txt
# containing the current interpreter/llvm rev number.
# If the current rev is different from the stored rev, force a make
# in the LLVM tree. Script returns 1 when make has to be forced, 0 otherwise.
# Exit status is always 0.

ret=0
dir=
dotsvn=".svn"
svn=svn
if [ $# = 1 ]; then
   if [ -x /bin/cygpath ]; then
      dir=`cygpath -u $1`
      svn=/usr/bin/svn
   else
      dir=$1
   fi
   dotsvn="$dir/.svn"
fi

# if we don't see the .svn directory, just return
if test ! -d $dotsvn; then
   echo $ret
   exit 0;
fi

OUT=interpreter/llvm/obj/llvmrev.txt
revold=
rev=`$svn info $dir | awk '/Last Changed Rev:/ { print $4 }'`

if [ -f $OUT ]; then
   revold=`cat $OUT`
   if [ $rev -ne $revold ]; then
      ret=1
   fi
else
   ret=1
fi

[ -d `dirname $OUT` ] || mkdir -p `dirname $OUT`
echo $rev > $OUT
echo $ret

exit 0
