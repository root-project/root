#!/bin/sh
rm -f CMakeCache.txt
/usr/local/bin/cmake  -DCMAKE_BUILD_TYPE="Release" -Ddataframe="ON" /home/runner/work/root/root 
