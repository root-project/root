
nmake CFG="long - Win32 Release" %1 %2 %3 %4 %5 %6
del long.dll
move Release\long.dll .\long.dll

