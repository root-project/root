#! /bin/sh

# Put dlls into bin/, symlinking them to lib/, and create
# a symlinked import archive .dll.a in lib/.

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
   *) args="$args $1" ;;
   esac
   shift
done

# 
g++ $args \
  && ( if [ "$isdll" != "0" ]; then \
          ln -sf ../bin/$dllbase $dllname; \
          ln -sf ../bin/$dllbase $dllname.a; \
       fi )

exit $?
