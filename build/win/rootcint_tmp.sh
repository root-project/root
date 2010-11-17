#! /bin/sh

# Convert /cygdrive/ and /home/ to windows-style paths.

args=core/utils/src/rootcint_tmp.exe

while [ "$1" != "" ]; do
   case "$1" in
   -I*) narg=`echo $1 | sed -e s/-I//`; args="$args -I`cygpath -m -- $narg`" ;;
   /*) args="$args `cygpath -m -- $1`" ;;
   *) args="$args $1" ;;
   esac
   shift
done

$args
stat=$?

exit $stat
