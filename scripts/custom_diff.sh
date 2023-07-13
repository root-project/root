#!/usr/bin/env bash

FILECMP="$1"
FILEOUT="$2"

prefix1='/tmp/roottest_log_fifo_'
prefix2='/tmp/roottest_ref_fifo_'

# if $RANDOM does not exist, use /dev/random
if [ -z $RANDOM ]; then
  RANDOM1=`od -An -N2 -i /dev/random`    
  RANDOM2=`od -An -N2 -i /dev/random`    

  # remove leading whitespaces
  RANDOM1=`echo $RANDOM1`    
  RANDOM2=`echo $RANDOM2`    
else
  RANDOM1=$RANDOM
  RANDOM2=$RANDOM
fi

fifo1=$prefix1$RANDOM1
fifo2=$prefix2$RANDOM2

mkfifo "$fifo1"
mkfifo "$fifo2"

grep -v Processing "$FILECMP" > "$fifo1" &
grep -v Processing "$FILEOUT" > "$fifo2" &

diff -u -w "$fifo2" "$fifo1"

rc=$?

rm "$fifo1" "$fifo2"

exit $rc
