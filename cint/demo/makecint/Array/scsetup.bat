REM # makecint demonstration for Symantec C++ 7.2
REM # Array.cxx as archived library

move Array.C Array.cxx
move Fundamen.h Fundament.h
move Fundamen.C Fundament.cxx
copy ..\Complex\Complex.cxx Complex.cxx
copy ..\Complex\Complex.C Complex.cxx
copy ..\Complex\Complex.h Complex.h

REM # Create Makefile
makecint -mk Makefile -o Array -H Fundament.h Array.h -C++ Fundament.cxx Array.cxx Complex.cxx

REM # Compile
smake clean
smake

REM # Test
cint -I../Complex ../Complex/Complex.cxx Fundament.cxx Array.cxx test.C
Array test.C
