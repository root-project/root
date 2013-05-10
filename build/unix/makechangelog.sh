#! /bin/sh

GIT2CL=build/unix/git2cl.pl

echo ""
echo "Generating README/ChangeLog from Git logs..."
echo ""

# Generate ChangeLog from version v5-27-01 till now
LC_ALL=C git log --pretty --numstat --summary 04563f7356865cf75d5fede74cdc7d12b8763779..HEAD | $GIT2CL > README/ChangeLog

exit 0
