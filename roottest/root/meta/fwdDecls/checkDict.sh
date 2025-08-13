#! /usr/bin/env sh

genreflex clingFwdDeclExample.h -s clingFwdDeclExample_selection.xml --quiet
cat clingFwdDeclExample_rflx.cpp | grep "class "|grep -v "//"
