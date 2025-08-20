#!/bin/sh
rm -f CMakeCache.txt
/usr/local/bin/cmake  -DCMAKE_BUILD_TYPE="Release" -Ddataframe="ON" /home/runner/work/root/root 
/usr/local/bin/cmake  -DCMAKE_BUILD_TYPE="Release" -Dbuiltin_freetype="ON" -Ddataframe="ON" -Dx11="OFF" /home/runner/work/root/root
/usr/local/bin/cmake  -DCMAKE_BUILD_TYPE="Release" -Dbuiltin_freetype="ON" -Ddataframe="ON" -Dfail-on-missing="OFF" -Dx11="OFF" -Dxrootd="OFF" /home/runner/work/root/root
