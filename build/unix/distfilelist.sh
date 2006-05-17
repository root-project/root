#!/bin/sh

# Script to produce list of files to be included in the binary distribution of ROOT.
# Called by makedist.sh.
#
# Axel, 2006-05-16

# $1 contains dir to prepend to file names

# dir name to prepend (e.g. for tar) - make sure it ends on '/'
PREPENDDIR=`echo ${1}|sed 's,/$,,'`/
if [ "x${PREPENDDIR}" = "x/" ]; then
   PREPENDDIR=
else
   cd $PREPENDDIR || exit 1
fi


# mixture of files, wildcards, and directories
WILDCARDS="LICENSE README bin \
   include lib cint/MAKEINFO cint/include \
   cint/lib cint/stl tutorials/*.cxx tutorials/*.C \
   tutorials/*.h tutorials/*.dat tutorials/mlpHiggs.root \
   tutorials/gallery.root tutorials/galaxy.root \
   tutorials/stock.root tutorials/worldmap.jpg \
   tutorials/mditestbg.xpm tutorials/fore.xpm \
   tutorials/runcatalog.sql tutorials/*.py tutorials/*.rb \
   tutorials/saxexample.xml tutorials/person.xml \
   tutorials/person.dtd \
   test/*.cxx test/*.h test/Makefile* test/README \
   test/RootShower/*.h test/RootShower/*.cxx \
   test/RootShower/*.rc test/RootShower/*.ico \
   test/RootShower/*.png test/RootShower/Makefile \
   test/RootShower/anim test/RootShower/icons test/ProofBench \
   macros icons fonts etc include/rmain.cxx"

# expand wildcards, recursively add directories
FILES=
for wc in ${WILDCARDS}; do
   if [ -d "${wc}" ]; then
      FILES="${FILES} `find ${wc} -type f`"
   else
      FILES="${FILES} ${wc}"
   fi
done

# check whether we have a precompiled header, so we can keep it
HAVEPRECOMP=`echo ${FILES} | grep include/precompile.h`
if [ "x${HAVEPRECOMP}" != "x" ]; then
   HAVEPRECOMP=include/precompile.h
fi

# remove all files we don't want, put one file per line
echo `echo ${FILES} | tr ' ' '\n' | sed \
  -e 's,^include/precompile\..*$,,' \
  -e 's,^lib/.*\.dll$,,' \
  -e 's,^.*.cvsignore$,,' \
  -e 's,^.*/CVS/.*$,,' \
   | grep -v -e '^$'` ${HAVEPRECOMP} | tr ' ' '\n' | sort | uniq | sed -e 's,^,'${PREPENDDIR}','
