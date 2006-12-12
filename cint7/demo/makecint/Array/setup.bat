REM # makecint demonstration script for Visual C++ 4.0
REM # Array.cxx as archived library

move Array.C Array.cxx
move Fundamen.h Fundament.h
move Fundamen.C Fundament.cxx
move Fundament.C Fundament.cxx
copy ..\Complex\Complex.cxx Complex.cxx
copy ..\Complex\Complex.C Complex.cxx
copy ..\Complex\Complex.h Complex.h

REM # Create Makefile
makecint -mk Makefile -o Array -H Fundament.h Array.h -C++ Fundament.cxx Array.cxx Complex.cxx

REM # Compile
nmake /F Makefile CFG="Array - Win32 Release" clean
nmake /F Makefile CFG="Array - Win32 Release"
move Release\Array.exe Array.exe

REM # Test
cint -I../Complex ../Complex/Complex.cxx Fundament.cxx Array.cxx test.C
Array test.C
