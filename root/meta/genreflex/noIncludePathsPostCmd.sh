#! /bin/bash

grep genreflex noIncludePaths_rflx.cpp
test $? -eq 1
