#! /bin/sh

# Patch to create soname.dll.a archives and to use it 
# for symbol-providers (--no-whole-archive) for linking
# Also puts dlls to bin, symlinking them to lib

args=
isdll=0
while [ "$1" != "" ]; do
   case "$1" in
   -o) args="$args $1"; shift; 
       dllname="$1"; dllbase=`basename $1`; 
       if [ "`echo $dllname | sed 's{^lib/.*\.dll${{'`" != "$dllname" ]; then
	   isdll=1
	   args="$args bin/$dllbase" 
       else
	   args="$args $1" 
       fi ;;
   -Wl,--no-whole-archive) 
      found_no_whole_archive=yes;
      args="$args $1" ;;
   *) args="$args $1" ;;
   esac
   shift
done

ldversion=0`ld -V| head -n1 | sed 's/.* \([[:digit:]]*$\)/\1/' | egrep '[[:digit:]]+'`
if [ $ldversion -lt 20030404 ]; then 
    outimplib="-Wl,--out-implib,${dllname}.a"
fi

# 
g++ $outimplib $args \
  && ( if [ "$isdll" != "0" ]; then \
          ln -sf ../bin/$dllbase $dllname; \
       fi )

exit $?
