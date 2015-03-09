@echo off

rem Store info about in which git branch, what SHA1 and at what date/time we executed make.

set dir=""
set dotgit=".git"
if not "%1"=="" set dotgit="%1\.git"

rem if we don't see the .git directory, just return
if not exist %dotgit% exit /b 0

set OUT=".\etc\gitinfo.txt"

call git.exe --git-dir=%dotgit% describe --all > %OUT%
call git.exe --git-dir=%dotgit% describe --always >> %OUT%

for /F "usebackq tokens=1,2 delims==" %%i in (`wmic os get LocalDateTime /VALUE 2^>NUL`) do if '.%%i.'=='.LocalDateTime.' set ldt=%%j
set ldt=%ldt:~6,2% %ldt:~4,2% %ldt:~0,4%, %ldt:~8,2%:%ldt:~10,2%:%ldt:~12,2%
echo %ldt%>> %OUT%

