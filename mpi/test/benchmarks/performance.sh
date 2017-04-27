#!/bin/bash

rootmpi -np 2 -b reduce.C
rootmpi -np 4 -b reduce.C
rootmpi -np 8 -b reduce.C

rm -rf reduceall.root
hadd reduceall.root reduce2.root reduce4.root reduce8.root
root -l plotreduceall.C
