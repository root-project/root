#! /bin/sh

ROOTSRCDIR=$1
GIT2CL=$ROOTSRCDIR/build/unix/git2cl.pl

echo ""
echo "Generating README/ChangeLog from Git logs..."
echo ""

# Generate ChangeLog from version v5-33-01 till now
( cd $ROOTSRCDIR && LC_ALL=C git log --pretty --numstat --summary 82607481af2c5a2ece9a0d343fe795b00f3940d8..HEAD ) | $GIT2CL > README/ChangeLog

exit 0
