#! /bin/sh

# Simple interface to LINK, tansforming -o <exe> to -out:<exe> and unix
# pathnames to windows pathnames.

args=
dll=
while [ "$1" != "" ]; do
   arg=`cygpath -w -- $1`
   case "$arg" in
   -o) args="$args -out:"; shift
       dll="$1"
       args="$args`cygpath -w -- $1`" ;;
   *) args="$args $arg" ;;
   esac
   shift
done

link $args || exit $?
if [ "$dll" != "" -a -f $dll.manifest ]; then
   if [ "${dll%.dll}" == "$dll" ]
       then resourceID=1; # .exe
       else resourceID=2  #.dll
   fi
   mt -nologo -manifest $dll.manifest -outputresource:${dll}\;$resourceID
   rm $dll.manifest
fi

exit $?
