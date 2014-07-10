#!/usr/bin/env bash

# Generate the header file from the Store info about in which git branch, what SHA1 and at what date/time
# we executed make.

echo '#ifndef ROOT_RGITCOMMIT_H' > $1
echo '#define ROOT_RGITCOMMIT_H' >> $1
echo '#define ROOT_GIT_BRANCH "'`head -n 1 etc/gitinfo.txt | tail -n1`'"' >> $1
echo '#define ROOT_GIT_COMMIT "'`head -n 2 etc/gitinfo.txt | tail -n1`'"' >> $1
echo '#endif' >> $1

