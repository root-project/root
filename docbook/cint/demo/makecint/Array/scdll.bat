REM # makecint demonstration for Symantec C++ 7.2
REM # Array.cxx as Dynamic Link Library

move Array.C Array.cxx
move Fundamen.h Fundament.h
move Fundamen.C Fundament.cxx
copy ..\Complex\Complex.cxx Complex.cxx
copy ..\Complex\Complex.C Complex.cxx
copy ..\Complex\Complex.h Complex.h

REM # Create Makefile
makecint -mk Makefile -dl Array.dll -I../Complex -H Fundament.h Array.h -C++ Fundament.cxx Array.cxx Complex.cxx

REM # Compile
smake clean
smake

REM # Test
cint -I../Complex ../Complex/Complex.cxx Fundament.cxx Array.cxx test.C
cint Array.dll test.C
