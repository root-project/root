@echo off

rem Generate the header file from the Store info about in which git branch,
rem what SHA1 and at what date/time we executed make.

for /f "delims=" %%a in ('powershell.exe -command "& {Get-Content .\etc\gitinfo.txt | select -First 1}"') do set GIT_BRANCH=%%a
for /f "delims=" %%a in ('powershell.exe -command "& {Get-Content .\etc\gitinfo.txt | select -First 2 | select -Last 1}"') do set GIT_COMMIT=%%a

echo #ifndef ROOT_RGITCOMMIT_H > %1
echo #define ROOT_RGITCOMMIT_H >> %1
echo #define ROOT_GIT_BRANCH "%GIT_BRANCH%" >> %1
echo #define ROOT_GIT_COMMIT "%GIT_COMMIT%" >> %1
echo #endif >> %1
