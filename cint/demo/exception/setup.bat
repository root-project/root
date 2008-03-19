echo # First, compile everything
cl ehdemo.cxx -o ehdemo.exe
ehdemo.exe
del ehdemo.exe

echo # 2nd, compile ehdemo.h and interpret ehdemo.cxx
echo # This method is recommended.
makecint -mk Makefile -dl ehdemo.dll -H ehdemo.h
nmake -f Makefile CFG="ehdemo - Win32 Release" clean
nmake -f Makefile CFG="ehdemo - Win32 Release" 
move Release\ehdemo.dll ehdemo.dll
cint ehdemo.cxx

echo # 3nd, interpret everything
nmake -f Makefile CFG="ehdemo - Win32 Release" clean
cint ehdemo.cxx

