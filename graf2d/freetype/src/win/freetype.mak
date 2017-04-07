# Microsoft Developer Studio Generated NMAKE File, Based on freetype.dsp
!IF "$(CFG)" == ""
CFG=freetype - Win32 Debug Singlethreaded
!MESSAGE No configuration specified. Defaulting to freetype - Win32 Debug Singlethreaded.
!ENDIF

!IF "$(CFG)" != "freetype - Win32 Release" && "$(CFG)" != "freetype - Win32 Debug" && "$(CFG)" != "freetype - Win32 Debug Multithreaded" && "$(CFG)" != "freetype - Win32 Release Multithreaded" && "$(CFG)" != "freetype - Win32 Release Singlethreaded" && "$(CFG)" != "freetype - Win32 Debug Singlethreaded"
!MESSAGE Invalid configuration "$(CFG)" specified.
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE
!MESSAGE NMAKE /f "freetype.mak" CFG="freetype - Win32 Debug Singlethreaded"
!MESSAGE
!MESSAGE Possible choices for configuration are:
!MESSAGE
!MESSAGE "freetype - Win32 Release" (based on "Win32 (x86) Static Library")
!MESSAGE "freetype - Win32 Debug" (based on "Win32 (x86) Static Library")
!MESSAGE "freetype - Win32 Debug Multithreaded" (based on "Win32 (x86) Static Library")
!MESSAGE "freetype - Win32 Release Multithreaded" (based on "Win32 (x86) Static Library")
!MESSAGE "freetype - Win32 Release Singlethreaded" (based on "Win32 (x86) Static Library")
!MESSAGE "freetype - Win32 Debug Singlethreaded" (based on "Win32 (x86) Static Library")
!MESSAGE
!ERROR An invalid configuration is specified.
!ENDIF

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE
NULL=nul
!ENDIF

!IF  "$(CFG)" == "freetype - Win32 Release"
OUTDIR=.\..\..\..\objs\release
INTDIR=.\..\..\..\objs\release
CPP_SWITCHES=$(NMAKECXXFLAGS) /Z7 /O2 /I "..\\..\\..\\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT2_BUILD_LIBRARY" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c
!ELSEIF  "$(CFG)" == "freetype - Win32 Debug"
OUTDIR=.\..\..\..\objs\debug
INTDIR=.\..\..\..\objs\debug
CPP_SWITCHES=$(NMAKECXXFLAGS) /Z7 /Od /I "..\\..\\..\\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /D "FT2_BUILD_LIBRARY" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c
!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Multithreaded"
OUTDIR=.\..\..\..\objs\debug_mt
INTDIR=.\..\..\..\objs\debug_mt
CPP_SWITCHES=$(NMAKECXXFLAGS) /Z7 /Od /I "..\\..\\..\\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /D "FT2_BUILD_LIBRARY" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c
!ELSEIF  "$(CFG)" == "freetype - Win32 Release Multithreaded"
OUTDIR=.\..\..\..\objs\release_mt
INTDIR=.\..\..\..\objs\release_mt
CPP_SWITCHES=$(NMAKECXXFLAGS) /Z7 /O2 /I "..\\..\\..\\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT2_BUILD_LIBRARY" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c
!ELSEIF  "$(CFG)" == "freetype - Win32 Release Singlethreaded"
OUTDIR=.\..\..\..\objs\release_st
INTDIR=.\..\..\..\objs\release_st
CPP_SWITCHES=$(NMAKECXXFLAGS) /Z7 /O2 /I "..\\..\\..\\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT2_BUILD_LIBRARY" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c
!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Singlethreaded"
OUTDIR=.\..\..\..\objs\debug_st
INTDIR=.\..\..\..\objs\debug_st
CPP_SWITCHES=$(NMAKECXXFLAGS) /Z7 /Od /I "..\\..\\..\\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /D "FT2_BUILD_LIBRARY" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c
!ENDIF

LIB32_OBJS= \
	"$(INTDIR)\autofit.obj" \
	"$(INTDIR)\bdf.obj" \
	"$(INTDIR)\cff.obj" \
	"$(INTDIR)\ftbase.obj" \
	"$(INTDIR)\ftbbox.obj" \
	"$(INTDIR)\ftbdf.obj" \
	"$(INTDIR)\ftbitmap.obj" \
	"$(INTDIR)\ftcache.obj" \
	"$(INTDIR)\ftdebug.obj" \
	"$(INTDIR)\ftfstype.obj" \
	"$(INTDIR)\ftgasp.obj" \
	"$(INTDIR)\ftglyph.obj" \
	"$(INTDIR)\ftgxval.obj" \
	"$(INTDIR)\ftgzip.obj" \
	"$(INTDIR)\ftinit.obj" \
	"$(INTDIR)\ftlcdfil.obj" \
	"$(INTDIR)\ftlzw.obj" \
	"$(INTDIR)\ftmm.obj" \
	"$(INTDIR)\ftotval.obj" \
	"$(INTDIR)\ftpatent.obj" \
	"$(INTDIR)\ftpfr.obj" \
	"$(INTDIR)\ftstroke.obj" \
	"$(INTDIR)\ftsynth.obj" \
	"$(INTDIR)\ftsystem.obj" \
	"$(INTDIR)\fttype1.obj" \
	"$(INTDIR)\ftwinfnt.obj" \
	"$(INTDIR)\pcf.obj" \
	"$(INTDIR)\pfr.obj" \
	"$(INTDIR)\psaux.obj" \
	"$(INTDIR)\pshinter.obj" \
	"$(INTDIR)\psmodule.obj" \
	"$(INTDIR)\raster.obj" \
	"$(INTDIR)\sfnt.obj" \
	"$(INTDIR)\smooth.obj" \
	"$(INTDIR)\truetype.obj" \
	"$(INTDIR)\type1.obj" \
	"$(INTDIR)\type1cid.obj" \
	"$(INTDIR)\type42.obj" \
	"$(INTDIR)\winfnt.obj"

!IF  "$(CFG)" == "freetype - Win32 Release"

ALL : "..\..\..\objs\freetype261.lib"

CLEAN :
	-@erase /q "$(INTDIR)\*.obj" >nul 2>&1
	-@erase /q "$(INTDIR)\*.idb" >nul 2>&1
	-@erase /q "..\..\..\objs\freetype261.lib" >nul 2>&1

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP=cl.exe
CPP_PROJ=$(NMAKECXXFLAGS) /Z7 /O2 /I "..\\..\\..\\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT2_BUILD_LIBRARY" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c

.c{$(INTDIR)}.obj::
   $(CPP) -nologo @<<
   $(CPP_PROJ) $<
<<

.cpp{$(INTDIR)}.obj::
   $(CPP) -nologo @<<
   $(CPP_PROJ) $<
<<

.cxx{$(INTDIR)}.obj::
   $(CPP) -nologo @<<
   $(CPP_PROJ) $<
<<

.c{$(INTDIR)}.sbr::
   $(CPP) -nologo @<<
   $(CPP_PROJ) $<
<<

.cpp{$(INTDIR)}.sbr::
   $(CPP) -nologo @<<
   $(CPP_PROJ) $<
<<

.cxx{$(INTDIR)}.sbr::
   $(CPP) -nologo @<<
   $(CPP_PROJ) $<
<<

RSC=rc.exe
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\freetype.bsc"
BSC32_SBRS= \

LIB32=link.exe -lib
LIB32_FLAGS=/nologo /out:"..\..\..\objs\freetype261.lib"

"..\..\..\objs\freetype261.lib" : "$(OUTDIR)" $(DEF_FILE) $(LIB32_OBJS)
    $(LIB32) @<<
  $(LIB32_FLAGS) $(DEF_FLAGS) $(LIB32_OBJS)
<<

!ELSEIF  "$(CFG)" == "freetype - Win32 Debug"

ALL : "..\..\..\objs\freetype261_D.lib"

CLEAN :
	-@erase /q "$(INTDIR)\*.obj" >nul 2>&1
	-@erase /q "$(INTDIR)\*.idb" >nul 2>&1
	-@erase /q "..\..\..\objs\freetype261_D.lib" >nul 2>&1

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP=cl.exe
CPP_PROJ=$(NMAKECXXFLAGS) /Z7 /Od /I "..\\..\\..\\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /D "FT2_BUILD_LIBRARY" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c

.c{$(INTDIR)}.obj::
   $(CPP) -nologo @<<
   $(CPP_PROJ) $<
<<

.cpp{$(INTDIR)}.obj::
   $(CPP) -nologo @<<
   $(CPP_PROJ) $<
<<

.cxx{$(INTDIR)}.obj::
   $(CPP) -nologo @<<
   $(CPP_PROJ) $<
<<

.c{$(INTDIR)}.sbr::
   $(CPP) -nologo @<<
   $(CPP_PROJ) $<
<<

.cpp{$(INTDIR)}.sbr::
   $(CPP) -nologo @<<
   $(CPP_PROJ) $<
<<

.cxx{$(INTDIR)}.sbr::
   $(CPP) -nologo @<<
   $(CPP_PROJ) $<
<<

RSC=rc.exe
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\freetype.bsc"
BSC32_SBRS= \

LIB32=link.exe -lib
LIB32_FLAGS=/nologo /out:"..\..\..\objs\freetype261_D.lib"

"..\..\..\objs\freetype261_D.lib" : "$(OUTDIR)" $(DEF_FILE) $(LIB32_OBJS)
    $(LIB32) @<<
  $(LIB32_FLAGS) $(DEF_FLAGS) $(LIB32_OBJS)
<<

!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Multithreaded"

ALL : "..\..\..\objs\freetype261MT_D.lib"

CLEAN :
	-@erase /q "$(INTDIR)\*.obj" >nul 2>&1
	-@erase /q "$(INTDIR)\*.idb" >nul 2>&1
	-@erase /q "..\..\..\objs\freetype261MT_D.lib" >nul 2>&1

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP=cl.exe
CPP_PROJ=$(NMAKECXXFLAGS) /Z7 /Od /I "..\\..\\..\\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /D "FT2_BUILD_LIBRARY" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c

.c{$(INTDIR)}.obj::
   $(CPP) -nologo @<<
   $(CPP_PROJ) $<
<<

.cpp{$(INTDIR)}.obj::
   $(CPP) -nologo @<<
   $(CPP_PROJ) $<
<<

.cxx{$(INTDIR)}.obj::
   $(CPP) -nologo @<<
   $(CPP_PROJ) $<
<<

.c{$(INTDIR)}.sbr::
   $(CPP) -nologo @<<
   $(CPP_PROJ) $<
<<

.cpp{$(INTDIR)}.sbr::
   $(CPP) -nologo @<<
   $(CPP_PROJ) $<
<<

.cxx{$(INTDIR)}.sbr::
   $(CPP) -nologo @<<
   $(CPP_PROJ) $<
<<

RSC=rc.exe
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\freetype.bsc"
BSC32_SBRS= \

LIB32=link.exe -lib
LIB32_FLAGS=/nologo /out:"..\..\..\objs\freetype261MT_D.lib"

"..\..\..\objs\freetype261MT_D.lib" : "$(OUTDIR)" $(DEF_FILE) $(LIB32_OBJS)
    $(LIB32) @<<
  $(LIB32_FLAGS) $(DEF_FLAGS) $(LIB32_OBJS)
<<

!ELSEIF  "$(CFG)" == "freetype - Win32 Release Multithreaded"

ALL : "..\..\..\objs\freetype261MT.lib"

CLEAN :
	-@erase /q "$(INTDIR)\*.obj" >nul 2>&1
	-@erase /q "$(INTDIR)\*.idb" >nul 2>&1
	-@erase /q "..\..\..\objs\freetype261MT.lib" >nul 2>&1

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP=cl.exe
CPP_PROJ=$(NMAKECXXFLAGS) /Z7 /O2 /I "..\\..\\..\\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT2_BUILD_LIBRARY" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c

.c{$(INTDIR)}.obj::
   $(CPP) -nologo @<<
   $(CPP_PROJ) $<
<<

.cpp{$(INTDIR)}.obj::
   $(CPP) -nologo @<<
   $(CPP_PROJ) $<
<<

.cxx{$(INTDIR)}.obj::
   $(CPP) -nologo @<<
   $(CPP_PROJ) $<
<<

.c{$(INTDIR)}.sbr::
   $(CPP) -nologo @<<
   $(CPP_PROJ) $<
<<

.cpp{$(INTDIR)}.sbr::
   $(CPP) -nologo @<<
   $(CPP_PROJ) $<
<<

.cxx{$(INTDIR)}.sbr::
   $(CPP) -nologo @<<
   $(CPP_PROJ) $<
<<

RSC=rc.exe
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\freetype.bsc"
BSC32_SBRS= \

LIB32=link.exe -lib
LIB32_FLAGS=/nologo /out:"..\..\..\objs\freetype261MT.lib"

"..\..\..\objs\freetype261MT.lib" : "$(OUTDIR)" $(DEF_FILE) $(LIB32_OBJS)
    $(LIB32) @<<
  $(LIB32_FLAGS) $(DEF_FLAGS) $(LIB32_OBJS)
<<

!ELSEIF  "$(CFG)" == "freetype - Win32 Release Singlethreaded"

ALL : "..\..\..\objs\freetype261ST.lib"

CLEAN :
	-@erase /q "$(INTDIR)\*.obj" >nul 2>&1
	-@erase /q "$(INTDIR)\*.idb" >nul 2>&1
	-@erase /q "..\..\..\objs\freetype261ST.lib" >nul 2>&1

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP=cl.exe
CPP_PROJ=$(NMAKECXXFLAGS) /Z7 /O2 /I "..\\..\\..\\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT2_BUILD_LIBRARY" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c

.c{$(INTDIR)}.obj::
   $(CPP) -nologo @<<
   $(CPP_PROJ) $<
<<

.cpp{$(INTDIR)}.obj::
   $(CPP) -nologo @<<
   $(CPP_PROJ) $<
<<

.cxx{$(INTDIR)}.obj::
   $(CPP) -nologo @<<
   $(CPP_PROJ) $<
<<

.c{$(INTDIR)}.sbr::
   $(CPP) -nologo @<<
   $(CPP_PROJ) $<
<<

.cpp{$(INTDIR)}.sbr::
   $(CPP) -nologo @<<
   $(CPP_PROJ) $<
<<

.cxx{$(INTDIR)}.sbr::
   $(CPP) -nologo @<<
   $(CPP_PROJ) $<
<<

RSC=rc.exe
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\freetype.bsc"
BSC32_SBRS= \

LIB32=link.exe -lib
LIB32_FLAGS=/out:"..\..\..\objs\freetype261ST.lib"

"..\..\..\objs\freetype261ST.lib" : "$(OUTDIR)" $(DEF_FILE) $(LIB32_OBJS)
    $(LIB32) @<<
  $(LIB32_FLAGS) $(DEF_FLAGS) $(LIB32_OBJS)
<<

!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Singlethreaded"

ALL : "..\..\..\objs\freetype261ST_D.lib"

CLEAN :
	-@erase /q "$(INTDIR)\*.obj" >nul 2>&1
	-@erase /q "$(INTDIR)\*.idb" >nul 2>&1
	-@erase /q "..\..\..\objs\freetype261ST_D.lib" >nul 2>&1

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP=cl.exe
CPP_PROJ=$(NMAKECXXFLAGS) /Z7 /Od /I "..\\..\\..\\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /D "FT2_BUILD_LIBRARY" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c

.c{$(INTDIR)}.obj::
   $(CPP) -nologo @<<
   $(CPP_PROJ) $<
<<

.cpp{$(INTDIR)}.obj::
   $(CPP) -nologo @<<
   $(CPP_PROJ) $<
<<

.cxx{$(INTDIR)}.obj::
   $(CPP) -nologo @<<
   $(CPP_PROJ) $<
<<

.c{$(INTDIR)}.sbr::
   $(CPP) -nologo @<<
   $(CPP_PROJ) $<
<<

.cpp{$(INTDIR)}.sbr::
   $(CPP) -nologo @<<
   $(CPP_PROJ) $<
<<

.cxx{$(INTDIR)}.sbr::
   $(CPP) -nologo @<<
   $(CPP_PROJ) $<
<<

RSC=rc.exe
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\freetype.bsc"
BSC32_SBRS= \

LIB32=link.exe -lib
LIB32_FLAGS=/nologo /out:"..\..\..\objs\freetype261ST_D.lib"

"..\..\..\objs\freetype261ST_D.lib" : "$(OUTDIR)" $(DEF_FILE) $(LIB32_OBJS)
    $(LIB32) @<<
  $(LIB32_FLAGS) $(DEF_FLAGS) $(LIB32_OBJS)
<<

!ENDIF


!IF "$(NO_EXTERNAL_DEPS)" != "1"
!IF EXISTS("freetype.dep")
!INCLUDE "freetype.dep"
!ELSE
!MESSAGE Warning: cannot find "freetype.dep"
!ENDIF
!ENDIF

SOURCE=..\..\..\src\autofit\autofit.c
"$(INTDIR)\autofit.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)

SOURCE=..\..\..\src\bdf\bdf.c
"$(INTDIR)\bdf.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) -nologo @<<
  $(CPP_SWITCHES) $(SOURCE)
<<

SOURCE=..\..\..\src\cff\cff.c
"$(INTDIR)\cff.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) -nologo @<<
  $(CPP_SWITCHES) $(SOURCE)
<<

SOURCE=..\..\..\src\base\ftbase.c
"$(INTDIR)\ftbase.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) -nologo @<<
  $(CPP_SWITCHES) $(SOURCE)
<<

SOURCE=..\..\..\src\base\ftbbox.c
"$(INTDIR)\ftbbox.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)

SOURCE=..\..\..\src\base\ftbdf.c
"$(INTDIR)\ftbdf.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)

SOURCE=..\..\..\src\base\ftbitmap.c
"$(INTDIR)\ftbitmap.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)

SOURCE=..\..\..\src\cache\ftcache.c
"$(INTDIR)\ftcache.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) -nologo @<<
  $(CPP_SWITCHES) $(SOURCE)
<<

SOURCE=..\ftdebug.c
"$(INTDIR)\ftdebug.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) -nologo @<<
  $(CPP_SWITCHES) $(SOURCE)
<<

SOURCE=..\..\..\src\base\ftfstype.c
"$(INTDIR)\ftfstype.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) -nologo @<<
  $(CPP_SWITCHES) $(SOURCE)
<<

SOURCE=..\..\..\src\base\ftgasp.c
"$(INTDIR)\ftgasp.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)

SOURCE=..\..\..\src\base\ftglyph.c
"$(INTDIR)\ftglyph.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) -nologo @<<
  $(CPP_SWITCHES) $(SOURCE)
<<

SOURCE=..\..\..\src\base\ftgxval.c
"$(INTDIR)\ftgxval.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=..\..\..\src\gzip\ftgzip.c
"$(INTDIR)\ftgzip.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)

SOURCE=..\..\..\src\base\ftinit.c
"$(INTDIR)\ftinit.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) -nologo @<<
  $(CPP_SWITCHES) $(SOURCE)
<<

SOURCE=..\..\..\src\base\ftlcdfil.c
"$(INTDIR)\ftlcdfil.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) -nologo @<<
  $(CPP_SWITCHES) $(SOURCE)
<<

SOURCE=..\..\..\src\lzw\ftlzw.c
"$(INTDIR)\ftlzw.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)

SOURCE=..\..\..\src\base\ftmm.c
"$(INTDIR)\ftmm.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) -nologo @<<
  $(CPP_SWITCHES) $(SOURCE)
<<

SOURCE=..\..\..\src\base\ftotval.c
"$(INTDIR)\ftotval.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)

SOURCE=..\..\..\src\base\ftpatent.c
"$(INTDIR)\ftpatent.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) -nologo @<<
  $(CPP_SWITCHES) $(SOURCE)
<<

SOURCE=..\..\..\src\base\ftpfr.c
"$(INTDIR)\ftpfr.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)

SOURCE=..\..\..\src\base\ftstroke.c
"$(INTDIR)\ftstroke.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)

SOURCE=..\..\..\src\base\ftsynth.c
"$(INTDIR)\ftsynth.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)

SOURCE=..\..\..\src\base\ftsystem.c
"$(INTDIR)\ftsystem.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) -nologo @<<
  $(CPP_SWITCHES) $(SOURCE)
<<

SOURCE=..\..\..\src\base\fttype1.c
"$(INTDIR)\fttype1.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)

SOURCE=..\..\..\src\base\ftwinfnt.c
"$(INTDIR)\ftwinfnt.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)

SOURCE=..\..\..\src\pcf\pcf.c
"$(INTDIR)\pcf.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) -nologo @<<
  $(CPP_SWITCHES) $(SOURCE)
<<

SOURCE=..\..\..\src\pfr\pfr.c
"$(INTDIR)\pfr.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)

SOURCE=..\..\..\src\psaux\psaux.c
"$(INTDIR)\psaux.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) -nologo @<<
  $(CPP_SWITCHES) $(SOURCE)
<<

SOURCE=..\..\..\src\pshinter\pshinter.c
"$(INTDIR)\pshinter.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) -nologo @<<
  $(CPP_SWITCHES) $(SOURCE)
<<

SOURCE=..\..\..\src\psnames\psmodule.c
"$(INTDIR)\psmodule.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) -nologo @<<
  $(CPP_SWITCHES) $(SOURCE)
<<

SOURCE=..\..\..\src\raster\raster.c
"$(INTDIR)\raster.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) -nologo @<<
  $(CPP_SWITCHES) $(SOURCE)
<<

SOURCE=..\..\..\src\sfnt\sfnt.c
"$(INTDIR)\sfnt.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) -nologo @<<
  $(CPP_SWITCHES) $(SOURCE)
<<

SOURCE=..\..\..\src\smooth\smooth.c
"$(INTDIR)\smooth.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) -nologo @<<
  $(CPP_SWITCHES) $(SOURCE)
<<

SOURCE=..\..\..\src\truetype\truetype.c
"$(INTDIR)\truetype.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) -nologo @<<
  $(CPP_SWITCHES) $(SOURCE)
<<

SOURCE=..\..\..\src\type1\type1.c
"$(INTDIR)\type1.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) -nologo @<<
  $(CPP_SWITCHES) $(SOURCE)
<<

SOURCE=..\..\..\src\cid\type1cid.c
"$(INTDIR)\type1cid.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) -nologo @<<
  $(CPP_SWITCHES) $(SOURCE)
<<

SOURCE=..\..\..\src\type42\type42.c
"$(INTDIR)\type42.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) -nologo @<<
  $(CPP_SWITCHES) $(SOURCE)
<<

SOURCE=..\..\..\src\winfonts\winfnt.c
"$(INTDIR)\winfnt.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) -nologo @<<
  $(CPP_SWITCHES) $(SOURCE)
<<

