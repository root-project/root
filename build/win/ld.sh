#! /bin/sh

# Simple interface to LINK, tansforming -o <exe> to -out:<exe> and unix
# pathnames to windows pathnames.

args=

while [ "$1" != "" ]; do
   arg=`cygpath -w -- $1`
   case "$arg" in
   -o) args="$args -out:"; shift; args="$args`cygpath -w -- $1`" ;;
   *) args="$args $arg" ;;
   esac
   shift
done

link $args

exit $?
