
REM # makecint demonstration for Borland C++ Builder 3.0 
REM # Complex.cpp as Dynamic Link Library

move Complex.C Complex.cpp

REM # Create Makefile
makecint -mk Makefile -dl Complex.dll -H Complex.h -C++ Complex.cpp -cint -M0x10

REM # Compile
make.exe -f Makefile 

REM # Test
cint Complex.cpp test.C
cint Complex.dll test.C




