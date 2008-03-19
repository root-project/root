echo off
echo CINT:: This is very slow. CINT interpreter interprets itself
echo on
%CINTSYSDIR%\cint -I%CINTSYSDIR% -I%CINTSYSDIR%/src +P testmain.c %1 %2 %3 %4 %5
