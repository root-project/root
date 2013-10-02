#!/usr/bin/env python
#
# scripts to replace a string in a set  of a files
#

import sys, re, os




with open("out.txt", "wt") as out:
    for line in open("arithmetics.cpp"):
        out.write(line.replace('main', 'arithmetics'))
