#!/bin/sh

# Script to import new version of CINT in the ROOT CVS tree.
# Called by main Makefile. Assumes original CINT distribution
# is in $HOME/cint.
#
# Author: Fons Rademakers, 29/2/2000
# and Axel, 2006-04-07

# Specify with trailing "." to not include subdirs (CVS is always ignored)
IGNORE=".cvsignore .\# /\# README dl.exp Makefile .rej .pdb .manifest .~ .txt Makebcdict Makeapiold Makeapi HISTORY"
IGNOREROOT="cint/include/errno.h cint/include/float.h cint/include/limits.h cint/include/locale.h \
            cint/include/math.h cint/include/signal.h cint/include/stddef.h cint/include/stdio.h \
            cint/include/stdlib.h cint/include/time.h cint/Module.mk cint/lib/ipc/ipcif.h.old"
IGNORECINT="cint/demo/ cint/doc/ cint/glob /cint/src/cbstrm.cpp cint/test/ cint/platform cint/readline/ \
            cint/lib/cintocx/ cint/lib/WildCard/ cint/lib/wintcldl/ cint/lib/wintcldl83/ cint/malloc/ \
            cint/ARCHIVE cint/MAKEINFO cint/C2cxx cint/cxx2 cint/chmod.cxx cint/export cint/EXPOSE \
            cint/INSTALLBIN cint/removesrc.bat cint/setup cint/src/v6_dmy"

# files containing list of new and old files
ADDEDLIST=/tmp/cintadded
REMOVEDLIST=/tmp/cintremoved
[ -f $ADDEDLIST ] && rm $ADDEDLIST
[ -f $REMOVEDLIST ] && rm $REMOVEDLIST

if [ ! -d $HOME/cint ]; then
   echo "Cannot find original CINT directory: $HOME/cint"
   exit 1
fi

IMPORTCINT=1
[ "$1" == "export" ] && IMPORTCINT=0

function cint2root {
    RET="${1/$HOME\//}"
    RET="${RET/platform\/aixdlfcn/src}"
    if [ "${RET/.h/}" != "$RET" ]; then
		RET="$(echo $RET | sed 's,cint/src/,cint/inc/,')"
		[ "$RET" == "cint/inc/dlfcn.h" ] && RET=cint/inc/aixdlfcn.h
		[ "$RET" == "cint/G__ci.h" ] && RET=cint/inc/G__ci.h
    fi
}

function root2cint {
	RET="$(echo $1 | sed 's,cint/inc/,cint/src/,')"
    [ "$1" == cint/inc/aixdlfcn.h ] && RET=cint/platform/aixdlfcn/dlfcn.h
    [ "$1" == cint/src/dlfcn.c ] && RET=cint/platform/aixdlfcn/dlfcn.c
    [ "$1" == cint/inc/G__ci.h ] && RET=cint/G__ci.h
    RET="$HOME/$RET"
}

function checkfile {
	root2cint "$1"
	[ ! -f "$1"   ] && MISSINGR=1 || MISSINGR=0
	[ ! -f "$RET" ] && MISSINGC=1 || MISSINGC=0
	local ORDER="$1"" ""$RET"
	[ $IMPORTCINT == 0 ] && ORDER="$RET"" ""$1"
	[ $MISSINGR == 0 -a $MISSINGC == 0 ] \
	  && diff -p -u -I '^//[\$]Id: .*$' $ORDER > /tmp/cintdiff \
	  && RET=
}

ALLFILES=cint/inc/G__ci.h

for f in $(find $HOME/cint -type f | grep -v CVS/ ); do
	cint2root "$f"
	for i in $IGNORECINT; do
		if [ "${RET/$i/}" != "$RET" ]; then
			RET=
			break
		fi
	done
	ALLFILES="$ALLFILES"" ""$RET"
done
for f in $(find cint -type f| grep -v CVS/ ); do
	for i in $IGNOREROOT; do
		if [ "${f/$i/}" != "$f" ]; then
			f=
			break
		fi
	done
	ALLFILES="$ALLFILES"" ""$f"
done

IGNORE="`echo $IGNORE | sed -e 's, ,[^[:space:]]*\\\)\\\|\\\([^[:space:]]*,g' -e 's,\.,\\\.,g'`"
IGNORE='\([^[:space:]]*'${IGNORE}'[^[:space:]]*\)'
ALLFILES="$(echo "$ALLFILES"|sort|uniq \
  | sed -e 's,cint/\(src/\|inc/\|include/\|\)\(done\|error\|iosenum.[^[:space:]]*\|systypes.h\|sys/types.h\),,g' \
        -e "s,$IGNORE,,g" \
		)"

PATCHDIR=.
[ $IMPORTCINT == 0 ] && PATCHDIR=$HOME
for f in $ALLFILES; do
	[ $IMPORTCINT == 0 ] && ( root2cint $f; f=$RET )
	checkfile $f
	REMOVED=$MISSINGC
	ADDED=$MISSINGR
	[ $IMPORTCINT == 0 ] && (REMOVED=$MISSINGR; ADDED=$MISSINGC)
	if [ $REMOVED == 1 ]; then
		if [ "${IGNOREROOT/$f/}" == "$IGNOREROOT" -a "$f" != "on" -a "$f" != "VisualBasic.lnk" ]; then
			echo $f >> $REMOVEDLIST
		fi
	else if [ $ADDED == 1 ]; then 
		if [ "${IGNORECINT/$f/}" == "$IGNORECINT" ]; then
			echo $f >> $ADDEDLIST
		fi
	else if [ "$RET" != "" ]; then 
		( cd $PATCHDIR \
		&& patch -p0 < /tmp/cintdiff \
		&& rm /tmp/cintdiff \
		|| ( echo Error patching $f\!; exit 1 ) )
	else rm /tmp/cintdiff
	fi; fi; fi
done

# copy man pages directly to man directory
if [ $IMPORTCINT == 1 ]; then
	diff $HOME/cint/doc/man1/cint.1 man/man1/ >/dev/null \
	  || cp $HOME/cint/doc/man1/cint.1 man/man1/
	diff $HOME/cint/doc/man1/makecint.1 man/man1/ >/dev/null\
	  || cp $HOME/cint/doc/man1/makecint.1 man/man1/
else
	diff man/man1/cint.1 $HOME/cint/doc/man1/ >/dev/null \
	  || cp man/man1/cint.1 $HOME/cint/doc/man1/
	diff man/man1/makecint.1 $HOME/cint/doc/man1/ >/dev/null \
	  || cp man/man1/makecint.1 $HOME/cint/doc/man1/
fi

if [ -f $ADDEDLIST ]; then
	if [ "`cat $ADDEDLIST` " != "" ]; then
		echo ""
		echo "Files that exist in Cint but not in ROOT (add to ROOT CVS):"
		cat $ADDEDLIST
		for NEWFILE in `cat $ADDEDLIST`
			do cp ~/$NEWFILE $NEWFILE
		done
	fi
fi
if [ -f $REMOVEDLIST ]; then 
	if [ "`cat $REMOVEDLIST `" != "" ]; then
		echo ""
		echo "Files that exist in ROOT but not in Cint (remove from ROOT CVS):"
		cat $REMOVEDLIST
	fi
fi

# cleanup
rm -rf $ADDEDLIST $REMOVEDLIST

exit 0
