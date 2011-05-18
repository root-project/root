rem nmake -f cintocx.mak CFG="cintocx - Win32 Release"
move Release\cintocx.ocx cintocx.ocx
regsvr32 cintocx.ocx

