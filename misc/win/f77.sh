#! /bin/sh

# Simple interface to FL32, tansforming -o <obj> to -Fo<obj> and unix
# pathnames to windows pathnames (fl32 does not access unix style names).
# When -link is specified the output option should be -out:<exe> and it
# should be at the end of the option list.

args=
dolink=no
link=

while [ "$1" != "" ]; do
   arg=$1
   case "$arg" in
   -optimize:*) args="$args $arg" ;;
   -link) link="$arg"; dolink=yes ;;
   -o) if [ "$dolink" = "yes" ]; then
          link="$link -out:"
          shift; link="$link`cygpath -w -- $1`"
       else
          args="$args -Fo"
          shift; args="$args`cygpath -w -- $1`"
       fi ;;
   -c) args="$args -c "; shift; args="$args`cygpath -w -- $1`" ;;
   -*) if [ "$dolink" = "yes" ]; then
          link="$link `cygpath -w -- $1`"
       else
          args="$args `cygpath -w -- $1`"
       fi ;;
   *) args="$args `cygpath -w -- $1`" ;;
   esac
   shift
done

if [ "$dolink" = "yes" ]; then
   fl32 $args $link
   stat=$?
   if [ $stat -eq 1 ]; then
      stat=0
   fi
else
   fl32 $args
   stat=$?
fi

rm -f *.rsp

exit $stat
