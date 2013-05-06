# Microsoft Developer Studio Generated NMAKE File, Based on asview.dsp
!IF "$(CFG)" == ""
CFG=asview - Win32 Debug
!MESSAGE No configuration specified. Defaulting to asview - Win32 Debug.
!ENDIF 

!IF "$(CFG)" != "asview - Win32 Release" && "$(CFG)" != "asview - Win32 Debug"
!MESSAGE Invalid configuration "$(CFG)" specified.
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "asview.mak" CFG="asview - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "asview - Win32 Release" (based on "Win32 (x86) Application")
!MESSAGE "asview - Win32 Debug" (based on "Win32 (x86) Application")
!MESSAGE 
!ERROR An invalid configuration is specified.
!ENDIF 

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE 
NULL=nul
!ENDIF 

CPP=cl.exe
MTL=midl.exe
RSC=rc.exe

!IF  "$(CFG)" == "asview - Win32 Release"

OUTDIR=.\Release
INTDIR=.\Release

!IF "$(RECURSE)" == "0" 

ALL : ".\asview.exe"

!ELSE 

ALL : "libAfterImage - Win32 Release" ".\asview.exe"

!ENDIF 

!IF "$(RECURSE)" == "1" 
CLEAN :"libAfterImage - Win32 ReleaseCLEAN" 
!ELSE 
CLEAN :
!ENDIF 
	-@erase "$(INTDIR)\asview.obj"
	-@erase "$(INTDIR)\asview.pch"
	-@erase "$(INTDIR)\StdAfx.obj"
	-@erase "$(INTDIR)\vc60.idb"
	-@erase ".\asview.exe"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP_PROJ=/nologo /ML /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /Fp"$(INTDIR)\asview.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 
MTL_PROJ=/nologo /D "NDEBUG" /mktyplib203 /win32 
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\asview.bsc" 
BSC32_SBRS= \
	
LINK32=link.exe
LINK32_FLAGS=kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:windows /incremental:no /pdb:"$(OUTDIR)\asview.pdb" /machine:I386 /nodefaultlib:"LIBCD" /out:"asview.exe" /editandcontinue:NO 
LINK32_OBJS= \
	"$(INTDIR)\asview.obj" \
	"$(INTDIR)\StdAfx.obj" \
	"..\libAfterImage.lib"

".\asview.exe" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

!ELSEIF  "$(CFG)" == "asview - Win32 Debug"

OUTDIR=.\Debug
INTDIR=.\Debug

!IF "$(RECURSE)" == "0" 

ALL : ".\asview.exe"

!ELSE 

ALL : "libAfterImage - Win32 Debug" ".\asview.exe"

!ENDIF 

!IF "$(RECURSE)" == "1" 
CLEAN :"libAfterImage - Win32 DebugCLEAN" 
!ELSE 
CLEAN :
!ENDIF 
	-@erase "$(INTDIR)\asview.obj"
	-@erase "$(INTDIR)\asview.pch"
	-@erase "$(INTDIR)\StdAfx.obj"
	-@erase "$(INTDIR)\vc60.idb"
	-@erase "$(INTDIR)\vc60.pdb"
	-@erase "$(OUTDIR)\asview.pdb"
	-@erase ".\asview.exe"
	-@erase ".\asview.ilk"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP_PROJ=/nologo /MLd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /Fp"$(INTDIR)\asview.pch" /Yu"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 
MTL_PROJ=/nologo /D "_DEBUG" /mktyplib203 /win32 
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\asview.bsc" 
BSC32_SBRS= \
	
LINK32=link.exe
LINK32_FLAGS=kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:windows /incremental:yes /pdb:"$(OUTDIR)\asview.pdb" /debug /machine:I386 /nodefaultlib:"LIBC" /out:"asview.exe" /pdbtype:sept 
LINK32_OBJS= \
	"$(INTDIR)\asview.obj" \
	"$(INTDIR)\StdAfx.obj" \
	"..\libAfterImage.lib"

".\asview.exe" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

!ENDIF 

.c{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cpp{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cxx{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.c{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cpp{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cxx{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<


!IF "$(NO_EXTERNAL_DEPS)" != "1"
!IF EXISTS("asview.dep")
!INCLUDE "asview.dep"
!ELSE 
!MESSAGE Warning: cannot find "asview.dep"
!ENDIF 
!ENDIF 


!IF "$(CFG)" == "asview - Win32 Release" || "$(CFG)" == "asview - Win32 Debug"
SOURCE=.\asview.cpp

"$(INTDIR)\asview.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\asview.pch"


SOURCE=.\StdAfx.cpp

!IF  "$(CFG)" == "asview - Win32 Release"

CPP_SWITCHES=/nologo /ML /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /Fp"$(INTDIR)\asview.pch" /Yc"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\StdAfx.obj"	"$(INTDIR)\asview.pch" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "asview - Win32 Debug"

CPP_SWITCHES=/nologo /MLd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /Fp"$(INTDIR)\asview.pch" /Yc"stdafx.h" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\StdAfx.obj"	"$(INTDIR)\asview.pch" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

!IF  "$(CFG)" == "asview - Win32 Release"

"libAfterImage - Win32 Release" : 
   cd "\devel\AfterStep\afterstep-stable\libAfterImage"
   $(MAKE) /$(MAKEFLAGS) /F ".\libAfterImage.mak" CFG="libAfterImage - Win32 Release" 
   cd ".\win32"

"libAfterImage - Win32 ReleaseCLEAN" : 
   cd "\devel\AfterStep\afterstep-stable\libAfterImage"
   $(MAKE) /$(MAKEFLAGS) /F ".\libAfterImage.mak" CFG="libAfterImage - Win32 Release" RECURSE=1 CLEAN 
   cd ".\win32"

!ELSEIF  "$(CFG)" == "asview - Win32 Debug"

"libAfterImage - Win32 Debug" : 
   cd "\devel\AfterStep\afterstep-stable\libAfterImage"
   $(MAKE) /$(MAKEFLAGS) /F ".\libAfterImage.mak" CFG="libAfterImage - Win32 Debug" 
   cd ".\win32"

"libAfterImage - Win32 DebugCLEAN" : 
   cd "\devel\AfterStep\afterstep-stable\libAfterImage"
   $(MAKE) /$(MAKEFLAGS) /F ".\libAfterImage.mak" CFG="libAfterImage - Win32 Debug" RECURSE=1 CLEAN 
   cd ".\win32"

!ENDIF 


!ENDIF 

