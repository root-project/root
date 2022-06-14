#! /bin/sh

# Simple interface to cl.exe, tansforming /cygdrive/... into "mixed" path.

args=

while [ "$1" != "" ]; do
   case "$1" in
   -I*) narg=`echo $1 | sed -e s/-I//`; args="$args -I`cygpath -m -- $narg`" ;;
   -c) args="$args -c "; shift; args="$args`cygpath -m -- $1`" ;;
   *) args="$args $1" ;;
   esac
   shift
done

cl.exe $args
stat=$?

if [ $stat -eq 126 ]; then
# cygwin causes spurious exit codes 126; just re-try.
   cl.exe $args
   stat=$?
fi

exit $stat
