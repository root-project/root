@echo off
rem Source this script to set up the ROOT build that this script is part of.
rem
rem Author: Axel Naumann, 10/07/2007

set OLDPATH=%CD%
set THIS=%0
set THIS=%THIS:~0,-12%.
cd /D %THIS%\..
set ROOTSYS=%CD%
cd /D %OLDPATH%
set PATH=%ROOTSYS%\bin;%PATH%
set CMAKE_PREFIX_PATH=%ROOTSYS%;%CMAKE_PREFIX_PATH%
set OLDPATH=
set THIS=
