
nmake CFG="win32api - Win32 Release" %1 %2 %3 %4 %5 %6
del win32api.dll
move Release\win32api.dll .\win32api.dll

