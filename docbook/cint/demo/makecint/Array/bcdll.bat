REM # makecint demonstration script for Borland C++ builder 3.0
REM # Array.cxx as Dynamic Link Library

move Array.C Array.cpp
move Fundamen.h Fundament.h
move Fundamen.C Fundament.cpp
move Fundament.C Fundament.cpp
copy ..\Complex\Complex.cxx Complex.cpp
copy ..\Complex\Complex.C Complex.cpp
copy ..\Complex\Complex.cpp Complex.cpp
copy ..\Complex\Complex.h Complex.h

REM # Create Makefile
makecint -mk Makefile -dl Array.dll -H Fundament.h Array.h -C++ Fundament.cpp Array.cpp Complex.cpp -cint -M0x10

REM # Compile
make.exe -f Makefile clean
make.exe -f Makefile 

REM # Test
cint -I../Complex ../Complex/Complex.cpp Fundament.cpp Array.cpp test.C
cint Array.dll test.C
