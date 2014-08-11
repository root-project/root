#! /bin/sh

# Put dlls into bin/, symlinking them to lib/, and create
# a symlinked import archive .dll.a in lib/.

CXX=`grep CXX config/Makefile.comp | sed 's,CXX[[:space:]]*=,,'`

args=
isdll=0
while [ "$1" != "" ]; do
   case "$1" in
   -o) args="$args $1"; shift;
       dllname="$1"; dllbase=`basename $1`;
       if [ "`echo $dllname | sed 's#^lib/.*\.dll$##'`" != "$dllname" ]; then
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
$CXX $args \
  && ( if [ "$isdll" != "0" ]; then \
          ln -sf $dllbase $dllname.a; \
          ln -sf ../bin/$dllbase $dllname; \
       fi )

exit $?
