#! /bin/sh

# Simple interface to CL, tansforming -o <obj> to -Fo<obj>

args=

while [ "$1" != "" ]; do
   case "$1" in
   -o) args="$args -Fo"; shift; args="$args$1" ;;
   *) args="$args $1" ;;
   esac
   shift
done

cl $args

exit $?
