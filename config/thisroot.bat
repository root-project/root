@echo off
rem Source this script to set up the ROOT build that this script is part of.
rem
rem Author: Axel Naumann, 10/07/2007

set OLDPATH=%CD%
set THIS=%0
set THIS=%THIS:~0,-12%.
cd %THIS%\..
set ROOTSYS=%CD%
cd %OLDPATH%
set PATH=%ROOTSYS%\bin;%PATH%
set OLDPATH=
set THIS=
