#! /bin/sh

# Patch to create soname.dll.a archives and to use it 
# for symbol-providers (--no-whole-archive) for linking
# cutting down mem usage by ld and build time
# Also copies dlls to bin

args=
while [ "$1" != "" ]; do
   case "$1" in
   -o) args="$args $1"; shift; dllname="$1"; dllbase=`basename $1`; args="$args bin/$dllbase" ;;
   -Wl,--no-whole-archive) 
      found_no_whole_archive=yes;
      args="$args $1" ;;
   *) args="$args $1" ;;
   esac
   shift
done

# 
g++ -Wl,--out-implib,${dllname}.a $args \
  && ( if [ "`echo $dllname | sed 's{^lib/.*\.dll${{'`" != "$dllname" ]; then \
          ln -sf ../bin/$dllbase $dllname; \
       fi ) \
  && chmod a+x bin/$dllbase

exit $?
