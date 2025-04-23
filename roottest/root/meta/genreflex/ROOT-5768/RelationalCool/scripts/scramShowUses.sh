#!/bin/sh
scram -debug b echo_INCLUDE | awk -f `dirname $0`/scramShowUses.awk
