#!/bin/bash

ROOT_DIR=$1

${ROOT_DIR}/test/eventexe 6 0 0 1 30 > log
cp Event.root event1.root
cp Event.root event2.root
