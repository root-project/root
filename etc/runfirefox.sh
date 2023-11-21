#!/bin/sh

# Helper script to run firefox and delete temporary profile at exit
profile=$1
shift

firefox=$1
shift

if [ "$profile" != "<dummy>" ]; then
   trap "rm -rf $profile; echo remove $profile at exit" 0 1 2 3 6
fi

args=

while [ "$1" != "" ]; do
   args="$args $1"
   shift
done

echo "Running $firefox $args"

$firefox $args
