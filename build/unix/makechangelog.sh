#! /bin/sh

CVS2CL=`which cvs2cl.pl`

dum=`echo $CVS2CL | grep "no cvs2cl.pl"`
stat=$?
if [ "$CVS2CL" = '' ] || [ $stat = 0 ]; then
   echo "cvs2cl.pl not found in PATH"
   return 1
fi

echo ""
echo "Generating README/ChangeLog from CVS logs..."
echo ""

$CVS2CL -f README/ChangeLog -W 10 -P -S --no-wrap

exit 0
