#!/usr/bin/env bash

# Store info about in which git branch, what SHA1 and at what date/time
# we executed make.

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
   exit 0;
fi

OUT=etc/gitinfo.txt

git describe --all > $OUT
git describe --always >> $OUT
date "+%b %d %Y, %H:%M:%S" >> $OUT

exit 0
