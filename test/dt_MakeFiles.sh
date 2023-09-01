#!/bin/sh

size=50
tracks=600
comp=0
action=1

if test $# -gt 0 ; then
    echo ajusting size to $1
    size=$1
fi


./Event $size $comp 0 $action $tracks
mv Event.root Event.new.split0.root

./Event $size $comp 1 $action $tracks
mv Event.root Event.new.split1.root

./Event $size $comp 9 $action $tracks
mv Event.root Event.new.split9.root

./Event $size $comp -1 $action $tracks
mv Event.root Event.old.streamed.root

./Event $size $comp -2 $action $tracks
mv Event.root Event.old.split.root

# Next step is to do something like
# root -q 'dt_MakeRef.C("Event.old.split.root");'
