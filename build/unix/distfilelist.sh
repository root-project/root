#!/bin/sh

# Script to produce list of files to be included in the
# binary distribution of ROOT.
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

# clean tutorials so we can include the entire directory
# (copy of code in main Makefile, so change there too if needed)
mv -f tutorials/gallery.root tutorials/gallery.root-
mv -f tutorials/mlp/mlpHiggs.root tutorials/mlp/mlpHiggs.root-
mv -f tutorials/quadp/stock.root tutorials/quadp/stock.root-
mv -f tutorials/proof/ntprndm.root tutorials/proof/ntprndm.root-
find tutorials -name "*.root" -exec rm -rf {} \; >/dev/null 2>&1;true
find tutorials -name "*.ps" -exec rm -rf {} \; >/dev/null 2>&1;true
find tutorials -path tutorials/doc -prune -o -name "*.gif" -exec rm -rf {} \; >/dev/null 2>&1;true
find tutorials -name "so_locations" -exec rm -rf {} \; >/dev/null 2>&1;true
find tutorials -name "pca.C" -exec rm -rf {} \; >/dev/null 2>&1;true
find tutorials -name "*.so" -exec rm -rf {} \; >/dev/null 2>&1;true
find tutorials -name "work.pc" -exec rm -rf {} \; >/dev/null 2>&1;true
find tutorials -name "work.pcl" -exec rm -rf {} \; >/dev/null 2>&1;true

mv -f tutorials/gallery.root- tutorials/gallery.root
mv -f tutorials/mlp/mlpHiggs.root- tutorials/mlp/mlpHiggs.root
mv -f tutorials/quadp/stock.root- tutorials/quadp/stock.root
mv -f tutorials/proof/ntprndm.root- tutorials/proof/ntprndm.root

# mixture of files, wildcards, and directories
WILDCARDS="LICENSE README bin \
   include lib man config/Makefile.comp config/Makefile.config \
   cint/cint/include tutorials \
   cint/cint/lib cint/cint/stl geom/gdml/*.py \
   test/*.cxx test/*.h test/Makefile* test/*.rootmap \
   test/*.C test/*.sh test/dt_Makefile test/*.ref \
   test/README test/*.txt test/*.xml \
   test/RootShower/*.h test/RootShower/*.cxx \
   test/RootShower/*.rc test/RootShower/*.ico \
   test/RootShower/*.png test/RootShower/Makefile* \
   test/RootShower/anim test/RootShower/icons \
   test/ProofBench test/RootIDE \
   tmva/test/*.gif tmva/test/*.png tmva/test/*.C tmva/test/README \
   macros icons fonts etc include/rmain.cxx"

# expand wildcards, recursively add directories
FILES=
for wc in ${WILDCARDS}; do
   if [ -d "${wc}" ]; then
      FILES="${FILES} `find ${wc} -type f -o -type l`"
   else
      FILES="${FILES} ${wc}"
   fi
done

# check whether we have a precompiled header, so we can keep it
HAVEPRECOMP=`echo ${FILES} | grep include/precompile.h`
if [ "x${HAVEPRECOMP}" != "x" ]; then
   HAVEPRECOMP=include/precompile.h
fi

FILES=`echo ${FILES} | tr ' ' '\n'`

ARCH=`grep '^ARCH' config/Makefile.config | sed 's,^ARCH.*:= ,,'`
if [ "x$ARCH" = "xwin32" ]; then
    FILES=`echo ${FILES} | tr ' ' '\n' | sed -e 's,^lib/.*\.dll$,,'`
fi
# remove all files we don't want, put one file per line
echo `echo ${FILES} | tr ' ' '\n' | sed \
  -e 's,^include/precompile\..*$,,' \
  -e 's,^.*.cvsignore$,,' \
  -e 's,^.*/CVS/.*$,,' \
  -e 's,^.*/.svn/.*$,,' \
  -e 's,^.*/.git/.*$,,' \
  -e 's,^.*/.*.dSYM/.*$,,' \
  -e 's,^cint/.*/G__c_.*$',, \
  -e 's,^cint/.*/G__cpp_.*$',, \
  -e 's,^cint/.*/rootcint_.*$',, \
   | grep -v '^$'` ${HAVEPRECOMP} | tr ' ' '\n' | sort | uniq | sed -e 's,^,'${PREPENDDIR}','
