#! /bin/sh

grep genreflex noIncludePaths_rflx.cpp
test $? -eq 1
