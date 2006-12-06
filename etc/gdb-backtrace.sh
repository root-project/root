#!/bin/sh

# This script is almost identical to /usr/bin/gstack.
# It is used by TUnixSystem::StackTrace() on Linux.

if test $# -ne 1; then
    echo "Usage: `basename $0 .sh` <process-id>" 1>&2
    exit 1
fi

if test ! -r /proc/$1; then
    echo "Process $1 not found." 1>&2
    exit 1
fi

# GDB doesn't allow "thread apply all bt" when the process isn't
# threaded; need to peek at the process to determine if that or the
# simpler "bt" should be used.

backtrace="bt"
if test -d /proc/$1/task ; then
    # Newer kernel; has a task/ directory.
    if test `ls /proc/$1/task | wc -l` -gt 1 2>/dev/null ; then
	backtrace="thread apply all bt"
    fi
elif test -f /proc/$1/maps ; then
    # Older kernel; go by it loading libpthread.
    if grep -e libpthread /proc/$1/maps > /dev/null 2>&1 ; then
	backtrace="thread apply all bt"
    fi
fi

GDB=${GDB:-gdb}

# Run GDB, strip out unwanted noise.
$GDB --quiet -nx /proc/$1/exe $1 <<EOF 2>&1 | 
$backtrace
EOF
egrep -v '^Reading |^Loaded |(gdb)' 
