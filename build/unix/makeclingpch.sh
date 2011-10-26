#!/bin/bash -x
clang=$1
shift # clang
"$@" || exit $?

# until bug http://llvm.org/bugs/show_bug.cgi?id=11236 is fixed:
exit 0

shift # rootcint_tmp
flags=""
ignorehdrs="VectorUtil_Cint.h"
while test ! "x$1" = "x"; do
    case $1 in
        -f) out=$2
            if `basename $out | grep rootcint_ > /dev/null`; then
                # No need for rootcint dicts
                exit 0
            fi
            pchstem=${out%.cxx}_dict
            allh=$pchstem.h
            pchout=lib/`basename $pchstem`.pcm;
            shift
            ;;
        -c) shift
            echo "// Cling dictionary include. Starts with vetos." > $allh
            echo "#define ROOT_Math_VectorUtil_Cint" >> $allh
            while test ! "x$1" = "x"; do
                case $1 in
                    -p) ;; # "call external PP", ignore
                    -*) flags="$flags $1" ;;
                    *)  echo '#include "'$1'"' | grep -v -i Linkdef.h | grep -v "VectorUtil_Cint.h" >> $allh ;;
                esac
                shift
            done
            $clang $flags -Xclang -emit-module -o$pchout -x c++ -c $allh || (ret=$?; rm $out; exit $ret)
            exit $?
            ;;
    esac
    shift   
done
