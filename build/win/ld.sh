#! /bin/sh

# Simple interface to LINK, tansforming -o <exe> to -out:<exe>

args=

while [ "$1" != "" ]; do
   case "$1" in
   -o) args="$args -out:"; shift; args="$args$1" ;;
   *) args="$args $1" ;;
   esac
   shift
done

link $args

exit $?
