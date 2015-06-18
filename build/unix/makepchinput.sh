#!/bin/sh
#
# Extract the input needed to build a PCH for the main (enabled) ROOT modules.
# Script takes as argument the source directory path, the set of enabled
# modules and extra headers (from cling) to be included in the PCH.
#
# Copyright (c) 2014 Rene Brun and Fons Rademakers
# Author: Axel Naumann <axel@cern.ch>, 2014-10-16

# Usage: $0 <root-srcdir> "module0 module1 ... moduleN" header0 header1 ... headerN -- cxxflag0 cxxflag1 ...

srcdir=$1
shift
modules=$1
shift

outdir=etc/dictpch
allheaders=$outdir/allHeaders.h
alllinkdefs=$outdir/allLinkDefs.h
cppflags=$outdir/allCppflags.txt

# Remove leftover files from old versions of this script.
rm -f include/allHeaders.h include/allHeaders.h.pch include/allLinkDef.h all.h cppflags.txt include/allLinkDef.h etc/allDict.cxx etc/allDict.cxx.h $cppflags.tmp $allheaders $alllinkdefs $cppflags

mkdir -p $outdir
rm -f $allheaders $alllinkdefs

# Here we include the list of c++11 stl headers
# From http://en.cppreference.com/w/cpp/header
# regex is removed until ROOT-7004 is fixed
stlHeaders="cstdlib csignal csetjmp cstdarg typeinfo typeindex type_traits bitset functional utility ctime chrono cstddef initializer_list tuple new memory scoped_allocator climits cfloat cstdint cinttypes limits exception stdexcept cassert system_error cerrno cctype cwctype cstring cwchar cuchar string array vector deque list forward_list set map unordered_set unordered_map stack queue algorithm iterator cmath complex valarray random numeric ratio cfenv iosfwd ios istream ostream iostream fstream sstream iomanip streambuf cstdio locale clocale codecvt atomic thread mutex future condition_variable ciso646 ccomplex ctgmath cstdalign cstdbool"

echo "// STL headers" >> $allheaders
for stlHeader in $stlHeaders; do
    echo '#if __has_include("'$stlHeader'")' >> $allheaders
    echo '#include <'$stlHeader'>' >> $allheaders
    echo '#endif' >> $allheaders
done

# Special case for regex
echo "// treat regex separately" >> $allheaders
echo '#if __has_include("regex") && !defined __APPLE__' >> $allheaders
echo '#include <regex>' >> $allheaders
echo '#endif' >> $allheaders

# treat this deprecated headers in a special way
stlDeprecatedHeaders="strstream"
echo "// STL Deprecated headers" >> $allheaders
echo "#define _BACKWARD_BACKWARD_WARNING_H" >> $allheaders
echo '#pragma clang diagnostic push' >> $allheaders
echo '#pragma GCC diagnostic ignored "-Wdeprecated"' >> $allheaders
for stlHeader in $stlDeprecatedHeaders; do
    echo '#if __has_include("'$stlHeader'")' >> $allheaders
    echo '#include <'$stlHeader'>' >> $allheaders
    echo '#endif' >> $allheaders
done
echo '#pragma clang diagnostic pop' >> $allheaders
echo '#undef _BACKWARD_BACKWARD_WARNING_H' >> $allheaders

while ! [ "x$1" = "x" -o "x$1" = "x--" ]; do
    echo '#include "'$1'"' >> $allheaders
    shift
done

if [ "x$1" = "x--" ]; then
    shift
fi

while ! [ "x$1" = "x" ]; do
    case $1 in
        -Wno*) echo "$1" >> $cppflags.tmp ;;
        -W*) ;;
        *) echo "$1" >> $cppflags.tmp ;;
    esac
    shift
done

for dict in `find $modules -name 'G__*.cxx' 2> /dev/null | grep -v /G__Cling.cxx  | grep -v core/metautils/src/G__std_`; do
    dirname=`dirname $dict`                   # to get foo/src
    dirname=`echo $dirname | sed -e 's,/src$,,' -e 's,^[.]/,,' ` # to get foo/

    case $dirname in
        graf2d/qt | math/fftw | math/foam | math/fumili | math/mlp | math/quadp | math/splot | math/unuran | math/vc | math/vdt) continue;;

        interpreter/* | core/* | io/io | net/net | math/* | hist/* | tree/* | graf2d/* | graf3d/ftgl | graf3d/g3d | graf3d/gl | gui/gui | gui/fitpanel | rootx | bindings/pyroot | roofit/* | tmva/* | main) ;;

        *) continue;;
    esac

    # Check if selmodules already contains the dirname.
    # Happens for instance for math/smatrix with its two (32bit and 64 bit)
    # dictionaries.
    if ! ( echo $selmodules | grep "$dirname " > /dev/null ); then
        selmodules="$selmodules$dirname "
    fi

    awk 'BEGIN{START=-1} /includePaths\[\] = {/, /^0$/ { if (START==-1) START=NR; else if ($0 != "0") { sub(/",/,"",$0); sub(/^"/,"-I",$0); print $0 } }' $dict >> $cppflags.tmp
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

    for f in `cd $srcdir; find $dirname/inc/ -name '*LinkDef*.h'`; do
        echo '#include "'$outdir/$f'"' >> $alllinkdefs
    done
done

# E.g. core's LinkDef includes clib/LinkDef, so just copy all LinkDefs.
for f in `cd $srcdir; find . -name '*LinkDef*.h'`; do
    mkdir -p $outdir/`dirname $f`
    cp $srcdir/$f $outdir/$f
done

cat $cppflags.tmp | sort | uniq | grep -v $srcdir | grep -v `pwd` > $cppflags

echo
echo Generating PCH for ${selmodules}
echo

