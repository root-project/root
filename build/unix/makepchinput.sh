#!/bin/sh
#
# Extract the input needed to build a PCH for the main (enabled) ROOT modules.
# Script takes as argument the source directory path, the set of enabled
# modules and extra headers (from cling) to be included in the PCH.
#
# Copyright (c) 2014 Rene Brun and Fons Rademakers
# Author: Axel Naumann <axel@cern.ch>, 2014-10-16

srcdir=$1
shift
modules=$1
shift

# Remove leftover files from old versions of this script.
rm -f include/allHeaders.h include/allHeaders.h.pch include/allLinkDef.h all.h cppflags.txt include/allLinkDef.h etc/allDict.cxx etc/allDict.cxx.h

outdir=etc/dictpch
allheaders=$outdir/allHeaders.h
alllinkdefs=$outdir/allLinkdefs.h
cppflags=$outdir/allCppflags.txt

mkdir -p $outdir
rm -f $allheaders $alllinkdefs

while ! [ "x$1" = "x" ]; do
    echo '#include "'$1'"' >> $allheaders
    shift
done

for dict in `find $modules -name 'G__*.cxx' 2> /dev/null | grep -v /G__Cling.cxx  | grep -v core/metautils/src/G__std_`; do
    dirname=`dirname $dict`                   # to get foo/src
    dirname=`echo $dirname | sed -e 's,/src$,,' -e 's,^[.]/,,' ` # to get foo/

    case $dirname in
        graf2d/qt | math/fftw | math/foam | math/fumili | math/mlp | math/quadp | math/splot | math/unuran | math/vc | math/vdt) continue;;

        interpreter/* | core/* | io/io | net/net | math/* | hist/* | tree/* | graf2d/* | graf3d/gl | gui/gui | gui/fitpanel | rootx | bindings/pyroot | roofit/* | tmva | main) ;;

        *) continue;;
    esac

    # Check if selmodules already contains the dirname.
    # Happens for instance for math/smatrix with its two (32bit and 64 bit)
    # dictionaries.
    if ! ( echo $selmodules | grep "$dirname " > /dev/null ); then
        selmodules="$selmodules$dirname "
    fi

    awk 'BEGIN{START=-1} /includePaths\[\] = {/, /^0$/ { if (START==-1) START=NR; else if ($0 != "0") { sub(/",/,"",$0); sub(/^"/,"-I",$0); print $0 } }' $dict >> $cppflags
    echo "// $dict" >> $allheaders
#     awk 'BEGIN{START=-1} /payloadCode =/, /^;$/ { if (START==-1) START=NR; else if ($1 != ";") { code=substr($0,2); sub(/\\n"/,"",code); print code } }' $dict >> $allheaders
    awk 'BEGIN{START=-1} /headers\[\] = {/, /^0$/ { if (START==-1) START=NR; else if ($0 != "0") { sub(/,/,"",$0); print "#include",$0 } }' $dict >> $allheaders

    if ! test "$dirname" = "`echo $dirname| sed 's,/qt,,'`"; then
        # something qt; undef emit afterwards
        cat <<EOF >> $allheaders
#ifdef emit
# undef emit
#endif
#ifdef signals
# undef signals
#endif
EOF
    elif ! test "$dirname" = "`echo $dirname| sed 's,net/ldap,,'`"; then
        # ldap; undef Debug afterwards
        cat <<EOF >> $allheaders
#ifdef Debug
# undef Debug
#endif
#ifdef GSL_SUCCESS
# undef GSL_SUCCESS
#endif
EOF
    fi

    find $srcdir/$dirname/inc/ -name '*LinkDef*.h' | \
        sed -e 's|^|#include "|' -e 's|$|"|' >> $alllinkdefs
done

echo
echo Generating PCH for ${selmodules}
echo
