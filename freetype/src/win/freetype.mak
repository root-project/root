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

ALL : "..\..\..\objs\freetype219.lib"


CLEAN :
	-@erase "$(INTDIR)\autohint.obj"
	-@erase "$(INTDIR)\bdf.obj"
	-@erase "$(INTDIR)\cff.obj"
	-@erase "$(INTDIR)\ftbase.obj"
	-@erase "$(INTDIR)\ftcache.obj"
	-@erase "$(INTDIR)\ftdebug.obj"
	-@erase "$(INTDIR)\ftglyph.obj"
	-@erase "$(INTDIR)\ftgzip.obj"
	-@erase "$(INTDIR)\ftinit.obj"
	-@erase "$(INTDIR)\ftlzw.obj"
	-@erase "$(INTDIR)\ftmm.obj"
	-@erase "$(INTDIR)\ftsystem.obj"
	-@erase "$(INTDIR)\pcf.obj"
	-@erase "$(INTDIR)\pfr.obj"
	-@erase "$(INTDIR)\psaux.obj"
	-@erase "$(INTDIR)\pshinter.obj"
	-@erase "$(INTDIR)\psmodule.obj"
	-@erase "$(INTDIR)\raster.obj"
	-@erase "$(INTDIR)\sfnt.obj"
	-@erase "$(INTDIR)\smooth.obj"
	-@erase "$(INTDIR)\truetype.obj"
	-@erase "$(INTDIR)\type1.obj"
	-@erase "$(INTDIR)\type1cid.obj"
	-@erase "$(INTDIR)\type42.obj"
	-@erase "$(INTDIR)\vc60.idb"
	-@erase "$(INTDIR)\vc60.pdb"
	-@erase "$(INTDIR)\winfnt.obj"
	-@erase "..\..\..\objs\freetype219.lib"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP=cl.exe
CPP_PROJ=/MD /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

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

RSC=rc.exe
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\freetype.bsc" 
BSC32_SBRS= \
	
LIB32=link.exe -lib
LIB32_FLAGS=/nologo /out:"..\..\..\objs\freetype219.lib" 
LIB32_OBJS= \
	"$(INTDIR)\autohint.obj" \
	"$(INTDIR)\bdf.obj" \
	"$(INTDIR)\cff.obj" \
	"$(INTDIR)\ftbase.obj" \
	"$(INTDIR)\ftcache.obj" \
	"$(INTDIR)\ftdebug.obj" \
	"$(INTDIR)\ftglyph.obj" \
	"$(INTDIR)\ftgzip.obj" \
	"$(INTDIR)\ftlzw.obj" \
	"$(INTDIR)\ftinit.obj" \
	"$(INTDIR)\ftmm.obj" \
	"$(INTDIR)\ftsystem.obj" \
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

"..\..\..\objs\freetype219.lib" : "$(OUTDIR)" $(DEF_FILE) $(LIB32_OBJS)
    $(LIB32) @<<
  $(LIB32_FLAGS) $(DEF_FLAGS) $(LIB32_OBJS)
<<

!ELSEIF  "$(CFG)" == "freetype - Win32 Debug"

OUTDIR=.\..\..\..\objs\debug
INTDIR=.\..\..\..\objs\debug

ALL : "..\..\..\objs\freetype219_D.lib"


CLEAN :
	-@erase "$(INTDIR)\autohint.obj"
	-@erase "$(INTDIR)\bdf.obj"
	-@erase "$(INTDIR)\cff.obj"
	-@erase "$(INTDIR)\ftbase.obj"
	-@erase "$(INTDIR)\ftcache.obj"
	-@erase "$(INTDIR)\ftdebug.obj"
	-@erase "$(INTDIR)\ftglyph.obj"
	-@erase "$(INTDIR)\ftgzip.obj"
	-@erase "$(INTDIR)\ftinit.obj"
	-@erase "$(INTDIR)\ftlzw.obj"
	-@erase "$(INTDIR)\ftmm.obj"
	-@erase "$(INTDIR)\ftsystem.obj"
	-@erase "$(INTDIR)\pcf.obj"
	-@erase "$(INTDIR)\pfr.obj"
	-@erase "$(INTDIR)\psaux.obj"
	-@erase "$(INTDIR)\pshinter.obj"
	-@erase "$(INTDIR)\psmodule.obj"
	-@erase "$(INTDIR)\raster.obj"
	-@erase "$(INTDIR)\sfnt.obj"
	-@erase "$(INTDIR)\smooth.obj"
	-@erase "$(INTDIR)\truetype.obj"
	-@erase "$(INTDIR)\type1.obj"
	-@erase "$(INTDIR)\type1cid.obj"
	-@erase "$(INTDIR)\type42.obj"
	-@erase "$(INTDIR)\vc60.idb"
	-@erase "$(INTDIR)\vc60.pdb"
	-@erase "$(INTDIR)\winfnt.obj"
	-@erase "..\..\..\objs\freetype219_D.lib"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP=cl.exe
CPP_PROJ=/MDd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

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

RSC=rc.exe
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\freetype.bsc" 
BSC32_SBRS= \
	
LIB32=link.exe -lib
LIB32_FLAGS=/nologo /out:"..\..\..\objs\freetype219_D.lib" 
LIB32_OBJS= \
	"$(INTDIR)\autohint.obj" \
	"$(INTDIR)\bdf.obj" \
	"$(INTDIR)\cff.obj" \
	"$(INTDIR)\ftbase.obj" \
	"$(INTDIR)\ftcache.obj" \
	"$(INTDIR)\ftdebug.obj" \
	"$(INTDIR)\ftglyph.obj" \
	"$(INTDIR)\ftgzip.obj" \
	"$(INTDIR)\ftlzw.obj" \
	"$(INTDIR)\ftinit.obj" \
	"$(INTDIR)\ftmm.obj" \
	"$(INTDIR)\ftsystem.obj" \
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

"..\..\..\objs\freetype219_D.lib" : "$(OUTDIR)" $(DEF_FILE) $(LIB32_OBJS)
    $(LIB32) @<<
  $(LIB32_FLAGS) $(DEF_FLAGS) $(LIB32_OBJS)
<<

!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Multithreaded"

OUTDIR=.\..\..\..\objs\debug_mt
INTDIR=.\..\..\..\objs\debug_mt

ALL : "..\..\..\objs\freetype219MT_D.lib"


CLEAN :
	-@erase "$(INTDIR)\autohint.obj"
	-@erase "$(INTDIR)\bdf.obj"
	-@erase "$(INTDIR)\cff.obj"
	-@erase "$(INTDIR)\ftbase.obj"
	-@erase "$(INTDIR)\ftcache.obj"
	-@erase "$(INTDIR)\ftdebug.obj"
	-@erase "$(INTDIR)\ftglyph.obj"
	-@erase "$(INTDIR)\ftgzip.obj"
	-@erase "$(INTDIR)\ftinit.obj"
	-@erase "$(INTDIR)\ftlzw.obj"
	-@erase "$(INTDIR)\ftmm.obj"
	-@erase "$(INTDIR)\ftsystem.obj"
	-@erase "$(INTDIR)\pcf.obj"
	-@erase "$(INTDIR)\pfr.obj"
	-@erase "$(INTDIR)\psaux.obj"
	-@erase "$(INTDIR)\pshinter.obj"
	-@erase "$(INTDIR)\psmodule.obj"
	-@erase "$(INTDIR)\raster.obj"
	-@erase "$(INTDIR)\sfnt.obj"
	-@erase "$(INTDIR)\smooth.obj"
	-@erase "$(INTDIR)\truetype.obj"
	-@erase "$(INTDIR)\type1.obj"
	-@erase "$(INTDIR)\type1cid.obj"
	-@erase "$(INTDIR)\type42.obj"
	-@erase "$(INTDIR)\vc60.idb"
	-@erase "$(INTDIR)\vc60.pdb"
	-@erase "$(INTDIR)\winfnt.obj"
	-@erase "..\..\..\objs\freetype219MT_D.lib"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP=cl.exe
CPP_PROJ=/MTd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

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

RSC=rc.exe
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\freetype.bsc" 
BSC32_SBRS= \
	
LIB32=link.exe -lib
LIB32_FLAGS=/nologo /out:"..\..\..\objs\freetype219MT_D.lib" 
LIB32_OBJS= \
	"$(INTDIR)\autohint.obj" \
	"$(INTDIR)\bdf.obj" \
	"$(INTDIR)\cff.obj" \
	"$(INTDIR)\ftbase.obj" \
	"$(INTDIR)\ftcache.obj" \
	"$(INTDIR)\ftdebug.obj" \
	"$(INTDIR)\ftglyph.obj" \
	"$(INTDIR)\ftgzip.obj" \
	"$(INTDIR)\ftlzw.obj" \
	"$(INTDIR)\ftinit.obj" \
	"$(INTDIR)\ftmm.obj" \
	"$(INTDIR)\ftsystem.obj" \
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

"..\..\..\objs\freetype219MT_D.lib" : "$(OUTDIR)" $(DEF_FILE) $(LIB32_OBJS)
    $(LIB32) @<<
  $(LIB32_FLAGS) $(DEF_FLAGS) $(LIB32_OBJS)
<<

!ELSEIF  "$(CFG)" == "freetype - Win32 Release Multithreaded"

OUTDIR=.\..\..\..\objs\release_mt
INTDIR=.\..\..\..\objs\release_mt

ALL : "..\..\..\objs\freetype219MT.lib"


CLEAN :
	-@erase "$(INTDIR)\autohint.obj"
	-@erase "$(INTDIR)\bdf.obj"
	-@erase "$(INTDIR)\cff.obj"
	-@erase "$(INTDIR)\ftbase.obj"
	-@erase "$(INTDIR)\ftcache.obj"
	-@erase "$(INTDIR)\ftdebug.obj"
	-@erase "$(INTDIR)\ftglyph.obj"
	-@erase "$(INTDIR)\ftgzip.obj"
	-@erase "$(INTDIR)\ftinit.obj"
	-@erase "$(INTDIR)\ftlzw.obj"
	-@erase "$(INTDIR)\ftmm.obj"
	-@erase "$(INTDIR)\ftsystem.obj"
	-@erase "$(INTDIR)\pcf.obj"
	-@erase "$(INTDIR)\pfr.obj"
	-@erase "$(INTDIR)\psaux.obj"
	-@erase "$(INTDIR)\pshinter.obj"
	-@erase "$(INTDIR)\psmodule.obj"
	-@erase "$(INTDIR)\raster.obj"
	-@erase "$(INTDIR)\sfnt.obj"
	-@erase "$(INTDIR)\smooth.obj"
	-@erase "$(INTDIR)\truetype.obj"
	-@erase "$(INTDIR)\type1.obj"
	-@erase "$(INTDIR)\type1cid.obj"
	-@erase "$(INTDIR)\type42.obj"
	-@erase "$(INTDIR)\vc60.idb"
	-@erase "$(INTDIR)\vc60.pdb"
	-@erase "$(INTDIR)\winfnt.obj"
	-@erase "..\..\..\objs\freetype219MT.lib"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP=cl.exe
CPP_PROJ=/MT /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

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

RSC=rc.exe
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\freetype.bsc" 
BSC32_SBRS= \
	
LIB32=link.exe -lib
LIB32_FLAGS=/nologo /out:"..\..\..\objs\freetype219MT.lib" 
LIB32_OBJS= \
	"$(INTDIR)\autohint.obj" \
	"$(INTDIR)\bdf.obj" \
	"$(INTDIR)\cff.obj" \
	"$(INTDIR)\ftbase.obj" \
	"$(INTDIR)\ftcache.obj" \
	"$(INTDIR)\ftdebug.obj" \
	"$(INTDIR)\ftglyph.obj" \
	"$(INTDIR)\ftgzip.obj" \
	"$(INTDIR)\ftlzw.obj" \
	"$(INTDIR)\ftinit.obj" \
	"$(INTDIR)\ftmm.obj" \
	"$(INTDIR)\ftsystem.obj" \
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

"..\..\..\objs\freetype219MT.lib" : "$(OUTDIR)" $(DEF_FILE) $(LIB32_OBJS)
    $(LIB32) @<<
  $(LIB32_FLAGS) $(DEF_FLAGS) $(LIB32_OBJS)
<<

!ELSEIF  "$(CFG)" == "freetype - Win32 Release Singlethreaded"

OUTDIR=.\..\..\..\objs\release_st
INTDIR=.\..\..\..\objs\release_st

ALL : "..\..\..\objs\freetype219ST.lib"


CLEAN :
	-@erase "$(INTDIR)\autohint.obj"
	-@erase "$(INTDIR)\bdf.obj"
	-@erase "$(INTDIR)\cff.obj"
	-@erase "$(INTDIR)\ftbase.obj"
	-@erase "$(INTDIR)\ftcache.obj"
	-@erase "$(INTDIR)\ftdebug.obj"
	-@erase "$(INTDIR)\ftglyph.obj"
	-@erase "$(INTDIR)\ftgzip.obj"
	-@erase "$(INTDIR)\ftinit.obj"
	-@erase "$(INTDIR)\ftlzw.obj"
	-@erase "$(INTDIR)\ftmm.obj"
	-@erase "$(INTDIR)\ftsystem.obj"
	-@erase "$(INTDIR)\pcf.obj"
	-@erase "$(INTDIR)\pfr.obj"
	-@erase "$(INTDIR)\psaux.obj"
	-@erase "$(INTDIR)\pshinter.obj"
	-@erase "$(INTDIR)\psmodule.obj"
	-@erase "$(INTDIR)\raster.obj"
	-@erase "$(INTDIR)\sfnt.obj"
	-@erase "$(INTDIR)\smooth.obj"
	-@erase "$(INTDIR)\truetype.obj"
	-@erase "$(INTDIR)\type1.obj"
	-@erase "$(INTDIR)\type1cid.obj"
	-@erase "$(INTDIR)\type42.obj"
	-@erase "$(INTDIR)\vc60.idb"
	-@erase "$(INTDIR)\vc60.pdb"
	-@erase "$(INTDIR)\winfnt.obj"
	-@erase "..\..\..\objs\freetype219ST.lib"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP=cl.exe
CPP_PROJ=/ML /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

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

RSC=rc.exe
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\freetype.bsc" 
BSC32_SBRS= \
	
LIB32=link.exe -lib
LIB32_FLAGS=/out:"..\..\..\objs\freetype219ST.lib" 
LIB32_OBJS= \
	"$(INTDIR)\autohint.obj" \
	"$(INTDIR)\bdf.obj" \
	"$(INTDIR)\cff.obj" \
	"$(INTDIR)\ftbase.obj" \
	"$(INTDIR)\ftcache.obj" \
	"$(INTDIR)\ftdebug.obj" \
	"$(INTDIR)\ftglyph.obj" \
	"$(INTDIR)\ftgzip.obj" \
	"$(INTDIR)\ftlzw.obj" \
	"$(INTDIR)\ftinit.obj" \
	"$(INTDIR)\ftmm.obj" \
	"$(INTDIR)\ftsystem.obj" \
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

"..\..\..\objs\freetype219ST.lib" : "$(OUTDIR)" $(DEF_FILE) $(LIB32_OBJS)
    $(LIB32) @<<
  $(LIB32_FLAGS) $(DEF_FLAGS) $(LIB32_OBJS)
<<

!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Singlethreaded"

OUTDIR=.\..\..\..\objs\debug_st
INTDIR=.\..\..\..\objs\debug_st

ALL : "..\..\..\objs\freetype219ST_D.lib"


CLEAN :
	-@erase "$(INTDIR)\autohint.obj"
	-@erase "$(INTDIR)\bdf.obj"
	-@erase "$(INTDIR)\cff.obj"
	-@erase "$(INTDIR)\ftbase.obj"
	-@erase "$(INTDIR)\ftcache.obj"
	-@erase "$(INTDIR)\ftdebug.obj"
	-@erase "$(INTDIR)\ftglyph.obj"
	-@erase "$(INTDIR)\ftgzip.obj"
	-@erase "$(INTDIR)\ftinit.obj"
	-@erase "$(INTDIR)\ftlzw.obj"
	-@erase "$(INTDIR)\ftmm.obj"
	-@erase "$(INTDIR)\ftsystem.obj"
	-@erase "$(INTDIR)\pcf.obj"
	-@erase "$(INTDIR)\pfr.obj"
	-@erase "$(INTDIR)\psaux.obj"
	-@erase "$(INTDIR)\pshinter.obj"
	-@erase "$(INTDIR)\psmodule.obj"
	-@erase "$(INTDIR)\raster.obj"
	-@erase "$(INTDIR)\sfnt.obj"
	-@erase "$(INTDIR)\smooth.obj"
	-@erase "$(INTDIR)\truetype.obj"
	-@erase "$(INTDIR)\type1.obj"
	-@erase "$(INTDIR)\type1cid.obj"
	-@erase "$(INTDIR)\type42.obj"
	-@erase "$(INTDIR)\vc60.idb"
	-@erase "$(INTDIR)\vc60.pdb"
	-@erase "$(INTDIR)\winfnt.obj"
	-@erase "..\..\..\objs\freetype219ST_D.lib"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP=cl.exe
CPP_PROJ=/MLd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

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

RSC=rc.exe
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\freetype.bsc" 
BSC32_SBRS= \
	
LIB32=link.exe -lib
LIB32_FLAGS=/nologo /out:"..\..\..\objs\freetype219ST_D.lib" 
LIB32_OBJS= \
	"$(INTDIR)\autohint.obj" \
	"$(INTDIR)\bdf.obj" \
	"$(INTDIR)\cff.obj" \
	"$(INTDIR)\ftbase.obj" \
	"$(INTDIR)\ftcache.obj" \
	"$(INTDIR)\ftdebug.obj" \
	"$(INTDIR)\ftglyph.obj" \
	"$(INTDIR)\ftgzip.obj" \
	"$(INTDIR)\ftlzw.obj" \
	"$(INTDIR)\ftinit.obj" \
	"$(INTDIR)\ftmm.obj" \
	"$(INTDIR)\ftsystem.obj" \
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

"..\..\..\objs\freetype219ST_D.lib" : "$(OUTDIR)" $(DEF_FILE) $(LIB32_OBJS)
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


!IF "$(CFG)" == "freetype - Win32 Release" || "$(CFG)" == "freetype - Win32 Debug" || "$(CFG)" == "freetype - Win32 Debug Multithreaded" || "$(CFG)" == "freetype - Win32 Release Multithreaded" || "$(CFG)" == "freetype - Win32 Release Singlethreaded" || "$(CFG)" == "freetype - Win32 Debug Singlethreaded"
SOURCE=..\..\..\src\autohint\autohint.c

!IF  "$(CFG)" == "freetype - Win32 Release"

CPP_SWITCHES=/MD /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\autohint.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug"

CPP_SWITCHES=/MDd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\autohint.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Multithreaded"

CPP_SWITCHES=/MTd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\autohint.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Multithreaded"

CPP_SWITCHES=/MT /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\autohint.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Singlethreaded"

CPP_SWITCHES=/ML /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\autohint.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Singlethreaded"

CPP_SWITCHES=/MLd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\autohint.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=..\..\..\src\bdf\bdf.c

!IF  "$(CFG)" == "freetype - Win32 Release"

CPP_SWITCHES=/MD /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\bdf.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug"

CPP_SWITCHES=/MDd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\bdf.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Multithreaded"

CPP_SWITCHES=/MTd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\bdf.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Multithreaded"

CPP_SWITCHES=/MT /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\bdf.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Singlethreaded"

CPP_SWITCHES=/ML /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\bdf.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Singlethreaded"

CPP_SWITCHES=/MLd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\bdf.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=..\..\..\src\cff\cff.c

!IF  "$(CFG)" == "freetype - Win32 Release"

CPP_SWITCHES=/MD /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\cff.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug"

CPP_SWITCHES=/MDd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\cff.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Multithreaded"

CPP_SWITCHES=/MTd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\cff.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Multithreaded"

CPP_SWITCHES=/MT /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\cff.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Singlethreaded"

CPP_SWITCHES=/ML /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\cff.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Singlethreaded"

CPP_SWITCHES=/MLd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\cff.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=..\..\..\src\base\ftbase.c

!IF  "$(CFG)" == "freetype - Win32 Release"

CPP_SWITCHES=/MD /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\ftbase.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug"

CPP_SWITCHES=/MDd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\ftbase.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Multithreaded"

CPP_SWITCHES=/MTd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\ftbase.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Multithreaded"

CPP_SWITCHES=/MT /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\ftbase.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Singlethreaded"

CPP_SWITCHES=/ML /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\ftbase.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Singlethreaded"

CPP_SWITCHES=/MLd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\ftbase.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=..\..\..\src\cache\ftcache.c

!IF  "$(CFG)" == "freetype - Win32 Release"

CPP_SWITCHES=/MD /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\ftcache.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug"

CPP_SWITCHES=/MDd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\ftcache.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Multithreaded"

CPP_SWITCHES=/MTd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\ftcache.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Multithreaded"

CPP_SWITCHES=/MT /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\ftcache.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Singlethreaded"

CPP_SWITCHES=/ML /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\ftcache.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Singlethreaded"

CPP_SWITCHES=/MLd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\ftcache.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=..\ftdebug.c

!IF  "$(CFG)" == "freetype - Win32 Release"

CPP_SWITCHES=/MD /Ze /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\ftdebug.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug"

CPP_SWITCHES=/MDd /Ze /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\ftdebug.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Multithreaded"

CPP_SWITCHES=/MTd /Ze /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\ftdebug.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Multithreaded"

CPP_SWITCHES=/MT /Ze /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\ftdebug.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Singlethreaded"

CPP_SWITCHES=/ML /Ze /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\ftdebug.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Singlethreaded"

CPP_SWITCHES=/MLd /Ze /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\ftdebug.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=..\..\..\src\base\ftglyph.c

!IF  "$(CFG)" == "freetype - Win32 Release"

CPP_SWITCHES=/MD /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\ftglyph.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug"

CPP_SWITCHES=/MDd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\ftglyph.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Multithreaded"

CPP_SWITCHES=/MTd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\ftglyph.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Multithreaded"

CPP_SWITCHES=/MT /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\ftglyph.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Singlethreaded"

CPP_SWITCHES=/ML /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\ftglyph.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Singlethreaded"

CPP_SWITCHES=/MLd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\ftglyph.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=..\..\..\src\gzip\ftgzip.c

"$(INTDIR)\ftgzip.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=..\..\..\src\base\ftinit.c

!IF  "$(CFG)" == "freetype - Win32 Release"

CPP_SWITCHES=/MD /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\ftinit.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug"

CPP_SWITCHES=/MDd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\ftinit.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Multithreaded"

CPP_SWITCHES=/MTd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\ftinit.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Multithreaded"

CPP_SWITCHES=/MT /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\ftinit.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Singlethreaded"

CPP_SWITCHES=/ML /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\ftinit.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Singlethreaded"

CPP_SWITCHES=/MLd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\ftinit.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=..\..\..\src\lzw\ftlzw.c

"$(INTDIR)\ftlzw.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=..\..\..\src\base\ftmm.c

!IF  "$(CFG)" == "freetype - Win32 Release"

CPP_SWITCHES=/MD /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\ftmm.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug"

CPP_SWITCHES=/MDd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\ftmm.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Multithreaded"

CPP_SWITCHES=/MTd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\ftmm.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Multithreaded"

CPP_SWITCHES=/MT /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\ftmm.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Singlethreaded"

CPP_SWITCHES=/ML /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\ftmm.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Singlethreaded"

CPP_SWITCHES=/MLd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\ftmm.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=..\..\..\src\base\ftsystem.c

!IF  "$(CFG)" == "freetype - Win32 Release"

CPP_SWITCHES=/MD /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\ftsystem.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug"

CPP_SWITCHES=/MDd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\ftsystem.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Multithreaded"

CPP_SWITCHES=/MTd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\ftsystem.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Multithreaded"

CPP_SWITCHES=/MT /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\ftsystem.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Singlethreaded"

CPP_SWITCHES=/ML /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\ftsystem.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Singlethreaded"

CPP_SWITCHES=/MLd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\ftsystem.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=..\..\..\src\pcf\pcf.c

!IF  "$(CFG)" == "freetype - Win32 Release"

CPP_SWITCHES=/MD /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\pcf.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug"

CPP_SWITCHES=/MDd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\pcf.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Multithreaded"

CPP_SWITCHES=/MTd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\pcf.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Multithreaded"

CPP_SWITCHES=/MT /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\pcf.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Singlethreaded"

CPP_SWITCHES=/ML /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\pcf.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Singlethreaded"

CPP_SWITCHES=/MLd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\pcf.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=..\..\..\src\pfr\pfr.c

"$(INTDIR)\pfr.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=..\..\..\src\psaux\psaux.c

!IF  "$(CFG)" == "freetype - Win32 Release"

CPP_SWITCHES=/MD /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\psaux.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug"

CPP_SWITCHES=/MDd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\psaux.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Multithreaded"

CPP_SWITCHES=/MTd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\psaux.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Multithreaded"

CPP_SWITCHES=/MT /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\psaux.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Singlethreaded"

CPP_SWITCHES=/ML /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\psaux.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Singlethreaded"

CPP_SWITCHES=/MLd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\psaux.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=..\..\..\src\pshinter\pshinter.c

!IF  "$(CFG)" == "freetype - Win32 Release"

CPP_SWITCHES=/MD /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\pshinter.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug"

CPP_SWITCHES=/MDd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\pshinter.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Multithreaded"

CPP_SWITCHES=/MTd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\pshinter.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Multithreaded"

CPP_SWITCHES=/MT /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\pshinter.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Singlethreaded"

CPP_SWITCHES=/ML /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\pshinter.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Singlethreaded"

CPP_SWITCHES=/MLd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\pshinter.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=..\..\..\src\psnames\psmodule.c

!IF  "$(CFG)" == "freetype - Win32 Release"

CPP_SWITCHES=/MD /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\psmodule.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug"

CPP_SWITCHES=/MDd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\psmodule.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Multithreaded"

CPP_SWITCHES=/MTd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\psmodule.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Multithreaded"

CPP_SWITCHES=/MT /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\psmodule.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Singlethreaded"

CPP_SWITCHES=/ML /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\psmodule.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Singlethreaded"

CPP_SWITCHES=/MLd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\psmodule.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=..\..\..\src\raster\raster.c

!IF  "$(CFG)" == "freetype - Win32 Release"

CPP_SWITCHES=/MD /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\raster.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug"

CPP_SWITCHES=/MDd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\raster.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Multithreaded"

CPP_SWITCHES=/MTd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\raster.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Multithreaded"

CPP_SWITCHES=/MT /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\raster.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Singlethreaded"

CPP_SWITCHES=/ML /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\raster.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Singlethreaded"

CPP_SWITCHES=/MLd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\raster.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=..\..\..\src\sfnt\sfnt.c

!IF  "$(CFG)" == "freetype - Win32 Release"

CPP_SWITCHES=/MD /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\sfnt.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug"

CPP_SWITCHES=/MDd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\sfnt.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Multithreaded"

CPP_SWITCHES=/MTd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\sfnt.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Multithreaded"

CPP_SWITCHES=/MT /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\sfnt.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Singlethreaded"

CPP_SWITCHES=/ML /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\sfnt.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Singlethreaded"

CPP_SWITCHES=/MLd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\sfnt.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=..\..\..\src\smooth\smooth.c

!IF  "$(CFG)" == "freetype - Win32 Release"

CPP_SWITCHES=/MD /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\smooth.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug"

CPP_SWITCHES=/MDd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\smooth.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Multithreaded"

CPP_SWITCHES=/MTd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\smooth.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Multithreaded"

CPP_SWITCHES=/MT /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\smooth.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Singlethreaded"

CPP_SWITCHES=/ML /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\smooth.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Singlethreaded"

CPP_SWITCHES=/MLd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\smooth.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=..\..\..\src\truetype\truetype.c

!IF  "$(CFG)" == "freetype - Win32 Release"

CPP_SWITCHES=/MD /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\truetype.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug"

CPP_SWITCHES=/MDd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\truetype.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Multithreaded"

CPP_SWITCHES=/MTd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\truetype.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Multithreaded"

CPP_SWITCHES=/MT /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\truetype.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Singlethreaded"

CPP_SWITCHES=/ML /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\truetype.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Singlethreaded"

CPP_SWITCHES=/MLd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\truetype.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=..\..\..\src\type1\type1.c

!IF  "$(CFG)" == "freetype - Win32 Release"

CPP_SWITCHES=/MD /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\type1.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug"

CPP_SWITCHES=/MDd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\type1.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Multithreaded"

CPP_SWITCHES=/MTd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\type1.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Multithreaded"

CPP_SWITCHES=/MT /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\type1.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Singlethreaded"

CPP_SWITCHES=/ML /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\type1.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Singlethreaded"

CPP_SWITCHES=/MLd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\type1.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=..\..\..\src\cid\type1cid.c

!IF  "$(CFG)" == "freetype - Win32 Release"

CPP_SWITCHES=/MD /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\type1cid.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug"

CPP_SWITCHES=/MDd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\type1cid.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Multithreaded"

CPP_SWITCHES=/MTd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\type1cid.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Multithreaded"

CPP_SWITCHES=/MT /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\type1cid.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Singlethreaded"

CPP_SWITCHES=/ML /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\type1cid.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Singlethreaded"

CPP_SWITCHES=/MLd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\type1cid.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=..\..\..\src\type42\type42.c

!IF  "$(CFG)" == "freetype - Win32 Release"

CPP_SWITCHES=/MD /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\type42.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug"

CPP_SWITCHES=/MDd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\type42.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Multithreaded"

CPP_SWITCHES=/MTd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\type42.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Multithreaded"

CPP_SWITCHES=/MT /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\type42.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Singlethreaded"

CPP_SWITCHES=/ML /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\type42.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Singlethreaded"

CPP_SWITCHES=/MLd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\type42.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=..\..\..\src\winfonts\winfnt.c

!IF  "$(CFG)" == "freetype - Win32 Release"

CPP_SWITCHES=/MD /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\winfnt.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug"

CPP_SWITCHES=/MDd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\winfnt.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Multithreaded"

CPP_SWITCHES=/MTd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\winfnt.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Multithreaded"

CPP_SWITCHES=/MT /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\winfnt.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Release Singlethreaded"

CPP_SWITCHES=/ML /Za /W4 /GX /Zi /O2 /I "..\..\..\include" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\winfnt.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "freetype - Win32 Debug Singlethreaded"

CPP_SWITCHES=/MLd /Za /W4 /GX /Zi /Od /I "..\..\..\include" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "FT_DEBUG_LEVEL_ERROR" /D "FT_DEBUG_LEVEL_TRACE" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\winfnt.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 


!ENDIF 

