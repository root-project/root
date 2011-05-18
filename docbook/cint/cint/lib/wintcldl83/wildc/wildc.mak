# Microsoft Developer Studio Generated NMAKE File, Based on wildc.dsp
!IF "$(CFG)" == ""
CFG=wildc - Win32 Release
!MESSAGE No configuration specified. Defaulting to wildc - Win32 Release.
!ENDIF 

!IF "$(CFG)" != "wildc - Win32 Release" && "$(CFG)" != "wildc - Win32 Debug"
!MESSAGE Invalid configuration "$(CFG)" specified.
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "wildc.mak" CFG="wildc - Win32 Release"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "wildc - Win32 Release" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE "wildc - Win32 Debug" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE 
!ERROR An invalid configuration is specified.
!ENDIF 

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE 
NULL=nul
!ENDIF 

!IF  "$(CFG)" == "wildc - Win32 Release"

OUTDIR=.\Release
INTDIR=.\Release
# Begin Custom Macros
OutDir=.\Release
# End Custom Macros

ALL : "$(OUTDIR)\wildc.dll"


CLEAN :
	-@erase "$(INTDIR)\cintlib.obj"
	-@erase "$(INTDIR)\vc60.idb"
	-@erase "$(INTDIR)\wildcIF.obj"
	-@erase "$(OUTDIR)\wildc.dll"
	-@erase "$(OUTDIR)\wildc.exp"
	-@erase "$(OUTDIR)\wildc.lib"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP=cl.exe
CPP_PROJ=/nologo /MT /w /W0 /GX /O2 /X /I "\cint" /I "\cint\src" /I "\cint\lib\vcstream" /I "\Program Files\tcl\include" /I "\cint\lib\stdstrct" /I "\Program Files\Microsoft Visual Studio\Vc98\include" /D "NDEBUG" /D "WIN32" /D "_WINDOWS" /D "G__WILDC" /D "G__WIN32" /D "G__VISUAL" /D "G__SHAREDLIB" /D "G__REDIRECTIO" /D "G__SPECIALSTDIO" /Fp"$(INTDIR)\wildc.pch" /YX /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

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

MTL=midl.exe
MTL_PROJ=/nologo /D "NDEBUG" /mktyplib203 /win32 
RSC=rc.exe
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\wildc.bsc" 
BSC32_SBRS= \
	
LINK32=link.exe
LINK32_FLAGS=kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib /nologo /subsystem:windows /dll /incremental:no /pdb:"$(OUTDIR)\wildc.pdb" /machine:I386 /def:"..\wildc.def" /out:"$(OUTDIR)\wildc.dll" /implib:"$(OUTDIR)\wildc.lib" 
DEF_FILE= \
	"..\wildc.def"
LINK32_OBJS= \
	"$(INTDIR)\cintlib.obj" \
	"$(INTDIR)\wildcIF.obj" \
	"..\..\..\Libcint.lib" \
	"..\..\..\..\Program Files\Tcl\lib\tk83.lib" \
	"..\..\..\..\Program Files\Tcl\lib\tcl83.lib"

"$(OUTDIR)\wildc.dll" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

!ELSEIF  "$(CFG)" == "wildc - Win32 Debug"

OUTDIR=.\Debug
INTDIR=.\Debug
# Begin Custom Macros
OutDir=.\Debug
# End Custom Macros

ALL : "$(OUTDIR)\wildc.dll"


CLEAN :
	-@erase "$(INTDIR)\cintlib.obj"
	-@erase "$(INTDIR)\vc60.idb"
	-@erase "$(INTDIR)\vc60.pdb"
	-@erase "$(INTDIR)\wildcIF.obj"
	-@erase "$(OUTDIR)\wildc.dll"
	-@erase "$(OUTDIR)\wildc.exp"
	-@erase "$(OUTDIR)\wildc.ilk"
	-@erase "$(OUTDIR)\wildc.lib"
	-@erase "$(OUTDIR)\wildc.pdb"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP=cl.exe
CPP_PROJ=/nologo /MTd /w /W0 /Gm /GX /ZI /Od /I "\cint" /I "\cint\src" /I "\cint\lib\vcstream" /I "\Program Files\tcl\include" /I "\cint\lib\stdstrct" /I "\Program Files\Microsoft Visual Studio\Vc98\include" /D "_DEBUG" /D "WIN32" /D "_WINDOWS" /D "G__WILDC" /D "G__WIN32" /D "G__VISUAL" /D "G__SHAREDLIB" /D "G__REDIRECTIO" /D "G__SPECIALSTDIO" /Fp"$(INTDIR)\wildc.pch" /YX /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

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

MTL=midl.exe
MTL_PROJ=/nologo /D "_DEBUG" /mktyplib203 /win32 
RSC=rc.exe
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\wildc.bsc" 
BSC32_SBRS= \
	
LINK32=link.exe
LINK32_FLAGS=kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib /nologo /subsystem:windows /dll /incremental:yes /pdb:"$(OUTDIR)\wildc.pdb" /debug /machine:I386 /def:"..\wildc.def" /out:"$(OUTDIR)\wildc.dll" /implib:"$(OUTDIR)\wildc.lib" 
DEF_FILE= \
	"..\wildc.def"
LINK32_OBJS= \
	"$(INTDIR)\cintlib.obj" \
	"$(INTDIR)\wildcIF.obj" \
	"..\..\..\Libcint.lib" \
	"..\..\..\..\Program Files\Tcl\lib\tk83.lib" \
	"..\..\..\..\Program Files\Tcl\lib\tcl83.lib"

"$(OUTDIR)\wildc.dll" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

!ENDIF 


!IF "$(NO_EXTERNAL_DEPS)" != "1"
!IF EXISTS("wildc.dep")
!INCLUDE "wildc.dep"
!ELSE 
!MESSAGE Warning: cannot find "wildc.dep"
!ENDIF 
!ENDIF 


!IF "$(CFG)" == "wildc - Win32 Release" || "$(CFG)" == "wildc - Win32 Debug"
SOURCE=..\cintlib.c

"$(INTDIR)\cintlib.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=..\wildcIF.c

"$(INTDIR)\wildcIF.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)



!ENDIF 

