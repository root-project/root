#!/bin/bash

FILECMP="$1"
FILEOUT="$2"

prefix1='/tmp/roottest_log_fifo_'
prefix2='/tmp/roottest_ref_fifo_'

fifo1=$prefix1$RANDOM
fifo2=$prefix2$RANDOM

mkfifo "$fifo1"
mkfifo "$fifo2"

grep -v Processing "$FILECMP" > "$fifo1" &
grep -v Processing "$FILEOUT" > "$fifo2" &

diff -u -w "$fifo2" "$fifo1"

rc=$?

rm "$fifo1" "$fifo2"

exit $rc
