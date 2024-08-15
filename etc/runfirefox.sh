#!/bin/sh

# Helper script to run firefox and delete temporary profile at exit
profile=$1
shift

firefox=$1
shift

if [ "$profile" != "<dummy>" ]; then
   trap "rm -rf $profile" 0 1 2 3 6
fi

args=

while [ "$1" != "" ]; do
   args="$args $1"
   shift
done

$firefox $args >/dev/null 2>/dev/null
