REM # makecint demonstration for Borland C++ Builder 3.0 
REM # DArray.cpp as Dynamic Link Library

move DArray.C DArray.cpp

REM # Create Makefile
makecint -mk Makefile -dl DArray.dll -H DArray.h -C++ DArray.cpp -cint -M0x10

REM # Compile
make.exe -f Makefile 

REM # Test
cint DArray.cpp test.C
cint DArray.dll test.C



