@echo off
SETLOCAL ENABLEEXTENSIONS
SETLOCAL ENABLEDELAYEDEXPANSION

set src=src\XrdVersion.hh.in
set dst=src\XrdVersion.hh
set str1=unknown
set str2=

if exist %dst% del %dst%

for /f "tokens=2* delims=," %%b in ('type VERSION_INFO^|findstr /c:"tag:"') do (
   set str2=%%b
)
if defined str2 (
   set str2=%str2:tag:=%
) else (
   for /f "tokens=2* delims=:" %%b in ('type VERSION_INFO^|findstr /c:"ShortHash:"') do (
      set str2=%%b
   )
)
set str2=%str2: =%

for /f "tokens=1,* delims=]" %%a in ('"%SystemRoot%\System32\find.exe /n /v "" "%src%""') do (
    if "%%b"=="" (
        echo.
    ) else (
        set line=%%b
        call echo.!line:%str1%=%str2%!
    )
)>>%dst%



