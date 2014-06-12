#!/bin/bash

FILEOUT="$1"
FILECMP="$2"

prefix1='/tmp/roottest_fifo1'
prefix2='/tmp/roottest_fifo2'

fifo1=$prefix1$RANDOM
fifo2=$prefix2$RANDOM

mkfifo "$fifo1"
mkfifo "$fifo2"

grep -v Processing "$FILEOUT" > "$fifo1" &
grep -v Processing "$FILECMP" > "$fifo2" &

diff -u -w "$fifo1" "$fifo2"

rc=$?

rm "$fifo1" "$fifo2"

exit $rc
