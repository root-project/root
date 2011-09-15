#! /bin/sh

# Simple interface to LINK, tansforming -o <exe> to -out:<exe> and unix
# pathnames to windows pathnames.

dir=`dirname $0`
args=
dll=
debug=
while [ "$1" != "" ]; do
   case "$1" in
   -o) args="$args -out:$2"
       dll="$2"
       shift ;;
   -debug) args="$args $1"
           debug=-DDEBUG;;
   *) args="$args $1" ;;
   esac
   shift
done

if [ "$dll" != "bin/rmkdepend.exe" -a \
     "$dll" != "bin/bindexplib.exe" -a \
     "$dll" != "bin/cint.exe" -a \
     "$dll" != "bin/makecint.exe" -a \
     "$dll" != "bin/rootcint.exe" -a \
     "$dll" != "bin/rlibmap.exe" -a \
     "$dll" != "bin/genmap.exe" -a \
     "$dll" != "cint/main/cint_tmp.exe" -a \
     "$dll" != "utils/src/rootcint_tmp.exe" -a \
     "$dll" != "bin/libCint.dll" -a \
     "$dll" != "bin/libReflex.dll" -a \
     "$dll" == "`echo $dll | sed 's,^cint/,,'`" -a \
     -r base/src/precompile.o ]; then
  args="$args base/src/precompile.o"
fi

if [ "$dll" != "" ]; then
   $dir/makeresource.sh "$dll" \
      && rc $debug -Iinclude -Fo"${dll}.res" "${dll}.rc" > /dev/null 2>&1
   if [ -r "${dll}.res" ]; then
      args="$args ${dll}.res"
   fi
   rm -f "${dll}.rc"
fi

WHICHLINK="`which cl.exe|sed 's,cl\.exe$,link.exe,'`"
"${WHICHLINK}" $args || exit $?

if [ "$dll" != "" -a -f $dll.manifest ]; then
   if [ "${dll%.dll}" == "$dll" ]
       then resourceID=1; # .exe
       else resourceID=2  #.dll
   fi
   mt -nologo -manifest $dll.manifest -outputresource:${dll}\;$resourceID
   rm $dll.manifest
fi

if [ "$dll" != "" ]; then
	rm -f "${dll}.res"
fi

exit 0
