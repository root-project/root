#!/usr/bin/env bash

if [ "x$1" = "x-h" -o "x$1" = "x--help" ]; then
    echo "Analyzes a roottest valgrind log file."
    echo ""
    echo "USAGE: `filename $0` [--leakoffset=<val>] [logfile]"
    echo "  --leakoffset: leakage bytes to ignore, default: 0"
    echo "  logfile: valgrind log from \"make valgrind\", default: newest in ./"
    exit
fi

if [ "x$1" != "x" -a "x${1/--leakoffset=/}" != "x$1" ]; then
    LEAKOFFSET=${1:13}
    shift
fi

# read in valgrind file - must be $1 or latest:
INFILE="$1"
if [ "x$1" = "x" ]; then
    INFILE="`ls -tr valgrind-*.log | tail -n1`"
fi
if [ "x$INFILE" = "x" ]; then
    echo 'ERROR: cannot find valgrind log in ./, and no log file specified.'
    exit 1
fi
if [ ! -r "$INFILE" ]; then
    echo 'ERROR: Cannot open input file '$INFILE
    exit 1
fi

# hold the currently active PIDs in the log.
# Active means that we are waiting for their summaries.
grep '==[[:digit:]]\+==' $INFILE | `dirname $0`/analyze_valgrind 
