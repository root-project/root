#! /bin/sh

# Simple interface to Qt's moc.exe, tansforming /cygdrive/... into "mixed" path.

args=

while [ "$1" != "" ]; do
   case "$1" in
   -o) args="$args $1" ;;
   *) args="$args `cygpath -m -- $1`" ;;
   esac
   shift
done

$args
stat=$?

exit $stat
