#! /bin/sh

CVS2CL=build/unix/cvs2cl.pl

echo ""
echo "Generating README/ChangeLog from CVS logs..."
echo ""

$CVS2CL -f README/ChangeLog -W 10 -P -S --no-wrap

rm -f README/ChangeLog.bak

exit 0
