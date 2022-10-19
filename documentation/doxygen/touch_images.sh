#!/bin/bash
FILES=$(grep -r macro_image $1 -l | sed 's!.*/!!' | sort -u | sed 's/$/.png/' | sed 's/^/_/')
echo "Touching images..."
for myfile in $FILES; do
    for i in {1..25} # 25 is the max number of pictures created by a macro, for double32.C
    do
        touch $2/pict$i$myfile &
    done
    wait
done
