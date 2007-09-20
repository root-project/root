#! /bin/sh

SVN2CL=build/unix/svn2cl.sh

echo ""
echo "Generating README/ChangeLog from SVN logs..."
echo ""

$SVN2CL -f README/ChangeLog

exit 0
