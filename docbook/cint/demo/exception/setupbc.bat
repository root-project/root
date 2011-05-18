echo # First, compile everything
bcc32 -P ehdemo.cxx -eehdemo.exe
ehdemo.exe
del ehdemo.exe

echo # 2nd, compile ehdemo.h and interpret ehdemo.cxx
echo # This method is recommended.
makecint -mk Makefile -dl ehdemo.dll -H ehdemo.h
make.exe -f Makefile clean
make.exe -f Makefile 
cint ehdemo.cxx

echo # 3nd, interpret everything
make.exe -f Makefile clean
cint ehdemo.cxx

