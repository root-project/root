#! /bin/sh
#
# A simple reconfigure script.
#
# Author: Axel Naumann
#
######################################################################

if [ ! -f config.status ]; then
  echo ""
  echo "Can't get config line from config.status."
  exit 1;
fi

confline=`cat config.status`

if [ "x$1" != "x" ]; then
  what=" because $1 has changed"
fi

if [ "x$2" != "x" ]; then
  srcdir="$2"
else
  srcdir="."
fi

echo ""
echo "Trying to reconfigure${what}."
echo "Using config statement:"
echo "$srcdir/configure $confline"
echo ""
$srcdir/configure $confline --nohowto || exit 1
echo "Reconfigure successful."
echo "If the build fails, please run $srcdir/configure again."
echo ""
test -e Makefile && sleep 1 && touch Makefile
