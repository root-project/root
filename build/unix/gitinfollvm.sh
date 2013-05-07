#!/usr/bin/env bash

# Check version of interpreter/llvm directory and if it is changed force
# a make in the LLVM tree. It creates the file
#    interpreter/llvm/obj/llvmrev.txt
# containing the current interpreter/llvm SHA1.
# If the current SHA1 is different from the stored SHA1, force a make
# in the LLVM tree. Script returns 1 when make has to be forced, 0 otherwise.
# Exit status is always 0.

ret=0
dir=
dotgit=".git"
git=git
if [ $# = 1 ]; then
   if [ -x /bin/cygpath ]; then
      dir=`cygpath -u $1`
      git=/usr/bin/git
   else
      dir=$1
   fi
   dotgit="$dir/.git"
fi

# if we don't see the .git directory, just return
if test ! -d $dotgit; then
   echo $ret
   exit 0;
fi

OUT=interpreter/llvm/obj/llvmrev.txt
revold=
rev=`$git --git-dir=$dotgit log -1 --pretty=format:"%h" -- interpreter/llvm`

if [ -f $OUT ]; then
   revold=`cat $OUT`
   if [ $rev != $revold ]; then
      ret=1
   fi
else
   ret=1
fi

[ -d `dirname $OUT` ] || mkdir -p `dirname $OUT`
echo $rev > $OUT
echo $ret

exit 0
