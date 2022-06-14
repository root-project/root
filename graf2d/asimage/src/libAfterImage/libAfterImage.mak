# Microsoft Developer Studio Generated NMAKE File, Based on libAfterImage.dsp
!IF "$(CFG)" == ""
CFG=libAfterImage - Win32 Debug
!MESSAGE No configuration specified. Defaulting to libAfterImage - Win32 Debug.
!ENDIF 

!IF "$(CFG)" != "libAfterImage - Win32 Release" && "$(CFG)" != "libAfterImage - Win32 Debug"
!MESSAGE Invalid configuration "$(CFG)" specified.
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "libAfterImage.mak" CFG="libAfterImage - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "libAfterImage - Win32 Release" (based on "Win32 (x86) Static Library")
!MESSAGE "libAfterImage - Win32 Debug" (based on "Win32 (x86) Static Library")
!MESSAGE 
!ERROR An invalid configuration is specified.
!ENDIF 

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE 
NULL=nul
!ENDIF 

CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "libAfterImage - Win32 Release"

OUTDIR=.\Release
INTDIR=.\Release

ALL : ".\libAfterImage.lib"


CLEAN :
	-@erase /q "$(INTDIR)"
	-@erase ".\libAfterImage.lib"

DISTCLEAN: CLEAN

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP_PROJ=$(FREETYPEDIRI) /nologo /W3 /O2 /D "WIN32" /D "NDEBUG" /D "NO_DEBUG_OUTPUT" /D "_MBCS" /D "_LIB" /Fp"$(INTDIR)\libAfterImage.pch" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c  $(NMAKECXXFLAGS)
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\libAfterImage.bsc" 
BSC32_SBRS= \
	
LIB32=link.exe -lib
LIB32_FLAGS=/nologo /out:"libAfterImage.lib" 
LIB32_OBJS= \
	"$(INTDIR)\png.obj" \
	"$(INTDIR)\pngerror.obj" \
	"$(INTDIR)\pngget.obj" \
	"$(INTDIR)\pngmem.obj" \
	"$(INTDIR)\pngpread.obj" \
	"$(INTDIR)\pngread.obj" \
	"$(INTDIR)\pngrio.obj" \
	"$(INTDIR)\pngrtran.obj" \
	"$(INTDIR)\pngrutil.obj" \
	"$(INTDIR)\pngset.obj" \
	"$(INTDIR)\pngtrans.obj" \
	"$(INTDIR)\pngwio.obj" \
	"$(INTDIR)\pngwrite.obj" \
	"$(INTDIR)\pngwtran.obj" \
	"$(INTDIR)\pngwutil.obj" \
	"$(INTDIR)\jcapimin.obj" \
	"$(INTDIR)\jcapistd.obj" \
	"$(INTDIR)\jccoefct.obj" \
	"$(INTDIR)\jccolor.obj" \
	"$(INTDIR)\jcdctmgr.obj" \
	"$(INTDIR)\jchuff.obj" \
	"$(INTDIR)\jcinit.obj" \
	"$(INTDIR)\jcmainct.obj" \
	"$(INTDIR)\jcmarker.obj" \
	"$(INTDIR)\jcmaster.obj" \
	"$(INTDIR)\jcomapi.obj" \
	"$(INTDIR)\jcparam.obj" \
	"$(INTDIR)\jcphuff.obj" \
	"$(INTDIR)\jcprepct.obj" \
	"$(INTDIR)\jcsample.obj" \
	"$(INTDIR)\jctrans.obj" \
	"$(INTDIR)\jdapimin.obj" \
	"$(INTDIR)\jdapistd.obj" \
	"$(INTDIR)\jdatadst.obj" \
	"$(INTDIR)\jdatasrc.obj" \
	"$(INTDIR)\jdcoefct.obj" \
	"$(INTDIR)\jdcolor.obj" \
	"$(INTDIR)\transupp.obj" \
	"$(INTDIR)\jaricom.obj" \
	"$(INTDIR)\jdarith.obj" \
	"$(INTDIR)\jcarith.obj" \
	"$(INTDIR)\jddctmgr.obj" \
	"$(INTDIR)\jdhuff.obj" \
	"$(INTDIR)\jdinput.obj" \
	"$(INTDIR)\jdmainct.obj" \
	"$(INTDIR)\jdmarker.obj" \
	"$(INTDIR)\jdmaster.obj" \
	"$(INTDIR)\jdmerge.obj" \
	"$(INTDIR)\jdpostct.obj" \
	"$(INTDIR)\jdsample.obj" \
	"$(INTDIR)\jdtrans.obj" \
	"$(INTDIR)\jerror.obj" \
	"$(INTDIR)\jfdctflt.obj" \
	"$(INTDIR)\jfdctfst.obj" \
	"$(INTDIR)\jfdctint.obj" \
	"$(INTDIR)\jidctflt.obj" \
	"$(INTDIR)\jidctfst.obj" \
	"$(INTDIR)\jidctint.obj" \
	"$(INTDIR)\jmemmgr.obj" \
	"$(INTDIR)\jmemnobs.obj" \
	"$(INTDIR)\jquant1.obj" \
	"$(INTDIR)\jquant2.obj" \
	"$(INTDIR)\jutils.obj" \
	"$(INTDIR)\dgif_lib.obj" \
	"$(INTDIR)\egif_lib.obj" \
	"$(INTDIR)\gif_err.obj" \
	"$(INTDIR)\gifalloc.obj" \
	"$(INTDIR)\gif_hash.obj" \
	"$(INTDIR)\afterbase.obj" \
	"$(INTDIR)\ascmap.obj" \
	"$(INTDIR)\asfont.obj" \
	"$(INTDIR)\asimage.obj" \
	"$(INTDIR)\asstorage.obj" \
	"$(INTDIR)\asimagexml.obj" \
	"$(INTDIR)\asvisual.obj" \
	"$(INTDIR)\blender.obj" \
	"$(INTDIR)\bmp.obj" \
	"$(INTDIR)\char2uni.obj" \
	"$(INTDIR)\export.obj" \
	"$(INTDIR)\import.obj" \
	"$(INTDIR)\pixmap.obj" \
	"$(INTDIR)\transform.obj" \
	"$(INTDIR)\ungif.obj" \
	"$(INTDIR)\xcf.obj" \
	"$(INTDIR)\ximage.obj" \
	"$(INTDIR)\xpm.obj" \
	"$(INTDIR)\draw.obj" \
	"$(INTDIR)\imencdec.obj" \
	"$(INTDIR)\scanline.obj"

".\libAfterImage.lib" : "$(OUTDIR)" $(DEF_FILE) $(LIB32_OBJS)
    $(LIB32) @<<
  $(LIB32_FLAGS) $(DEF_FLAGS) $(LIB32_OBJS)
<<

!ELSEIF  "$(CFG)" == "libAfterImage - Win32 Debug"

OUTDIR=.
INTDIR=.\win32\Debug
# Begin Custom Macros
OutDir=.
# End Custom Macros

ALL : "$(OUTDIR)\libAfterImage.lib"


CLEAN :
	-@erase /q "$(INTDIR)"
	-@erase "$(OUTDIR)\libAfterImage.lib"

DISTCLEAN: CLEAN

"$(INTDIR)" :
    if not exist "$(INTDIR)/$(NULL)" mkdir "$(INTDIR)"

CPP_PROJ=$(FREETYPEDIRI) /nologo /W3 /Gm /ZI /Od /D "WIN32" /D "_DEBUG" /D "NO_DEBUG_OUTPUT" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c $(NMAKECXXFLAGS)
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\libAfterImage.bsc" 
BSC32_SBRS= \
	
LIB32=link.exe -lib
LIB32_FLAGS=/nologo /out:"$(OUTDIR)\libAfterImage.lib" 
LIB32_OBJS= \
	"$(INTDIR)\png.obj" \
	"$(INTDIR)\pngerror.obj" \
	"$(INTDIR)\pngget.obj" \
	"$(INTDIR)\pngmem.obj" \
	"$(INTDIR)\pngpread.obj" \
	"$(INTDIR)\pngread.obj" \
	"$(INTDIR)\pngrio.obj" \
	"$(INTDIR)\pngrtran.obj" \
	"$(INTDIR)\pngrutil.obj" \
	"$(INTDIR)\pngset.obj" \
	"$(INTDIR)\pngtrans.obj" \
	"$(INTDIR)\pngwio.obj" \
	"$(INTDIR)\pngwrite.obj" \
	"$(INTDIR)\pngwtran.obj" \
	"$(INTDIR)\pngwutil.obj" \
	"$(INTDIR)\jcapimin.obj" \
	"$(INTDIR)\jcapistd.obj" \
	"$(INTDIR)\jccoefct.obj" \
	"$(INTDIR)\jccolor.obj" \
	"$(INTDIR)\jcdctmgr.obj" \
	"$(INTDIR)\jchuff.obj" \
	"$(INTDIR)\jcinit.obj" \
	"$(INTDIR)\jcmainct.obj" \
	"$(INTDIR)\jcmarker.obj" \
	"$(INTDIR)\jcmaster.obj" \
	"$(INTDIR)\jcomapi.obj" \
	"$(INTDIR)\jcparam.obj" \
	"$(INTDIR)\jcphuff.obj" \
	"$(INTDIR)\jcprepct.obj" \
	"$(INTDIR)\jcsample.obj" \
	"$(INTDIR)\jctrans.obj" \
	"$(INTDIR)\jdapimin.obj" \
	"$(INTDIR)\jdapistd.obj" \
	"$(INTDIR)\jdatadst.obj" \
	"$(INTDIR)\jdatasrc.obj" \
	"$(INTDIR)\jdcoefct.obj" \
	"$(INTDIR)\jdcolor.obj" \
	"$(INTDIR)\transupp.obj" \
	"$(INTDIR)\jaricom.obj" \
	"$(INTDIR)\jdarith.obj" \
	"$(INTDIR)\jcarith.obj" \
	"$(INTDIR)\jddctmgr.obj" \
	"$(INTDIR)\jdhuff.obj" \
	"$(INTDIR)\jdinput.obj" \
	"$(INTDIR)\jdmainct.obj" \
	"$(INTDIR)\jdmarker.obj" \
	"$(INTDIR)\jdmaster.obj" \
	"$(INTDIR)\jdmerge.obj" \
	"$(INTDIR)\jdpostct.obj" \
	"$(INTDIR)\jdsample.obj" \
	"$(INTDIR)\jdtrans.obj" \
	"$(INTDIR)\jerror.obj" \
	"$(INTDIR)\jfdctflt.obj" \
	"$(INTDIR)\jfdctfst.obj" \
	"$(INTDIR)\jfdctint.obj" \
	"$(INTDIR)\jidctflt.obj" \
	"$(INTDIR)\jidctfst.obj" \
	"$(INTDIR)\jidctint.obj" \
	"$(INTDIR)\jmemmgr.obj" \
	"$(INTDIR)\jmemnobs.obj" \
	"$(INTDIR)\jquant1.obj" \
	"$(INTDIR)\jquant2.obj" \
	"$(INTDIR)\jutils.obj" \
	"$(INTDIR)\dgif_lib.obj" \
	"$(INTDIR)\egif_lib.obj" \
	"$(INTDIR)\gif_err.obj" \
	"$(INTDIR)\gifalloc.obj" \
	"$(INTDIR)\gif_hash.obj" \
	"$(INTDIR)\afterbase.obj" \
	"$(INTDIR)\ascmap.obj" \
	"$(INTDIR)\asfont.obj" \
	"$(INTDIR)\asimage.obj" \
	"$(INTDIR)\asstorage.obj" \
	"$(INTDIR)\asimagexml.obj" \
	"$(INTDIR)\asvisual.obj" \
	"$(INTDIR)\blender.obj" \
	"$(INTDIR)\bmp.obj" \
	"$(INTDIR)\char2uni.obj" \
	"$(INTDIR)\export.obj" \
	"$(INTDIR)\import.obj" \
	"$(INTDIR)\pixmap.obj" \
	"$(INTDIR)\transform.obj" \
	"$(INTDIR)\ungif.obj" \
	"$(INTDIR)\xcf.obj" \
	"$(INTDIR)\ximage.obj" \
	"$(INTDIR)\xpm.obj" \
	"$(INTDIR)\draw.obj" \
	"$(INTDIR)\imencdec.obj" \
	"$(INTDIR)\scanline.obj"

"$(OUTDIR)\libAfterImage.lib" : "$(OUTDIR)" $(DEF_FILE) $(LIB32_OBJS)
    $(LIB32) @<<
  $(LIB32_FLAGS) $(DEF_FLAGS) $(LIB32_OBJS)
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
!IF EXISTS("libAfterImage.dep")
!INCLUDE "libAfterImage.dep"
!ELSE 
!MESSAGE Cannot find "libAfterImage.dep"
!ENDIF 
!ENDIF 


!IF "$(CFG)" == "libAfterImage - Win32 Release" || "$(CFG)" == "libAfterImage - Win32 Debug"
SOURCE=.\libpng\png.c

"$(INTDIR)\png.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libpng\pngerror.c

"$(INTDIR)\pngerror.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libpng\pngget.c

"$(INTDIR)\pngget.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libpng\pngmem.c

"$(INTDIR)\pngmem.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libpng\pngpread.c

"$(INTDIR)\pngpread.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libpng\pngread.c

"$(INTDIR)\pngread.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libpng\pngrio.c

"$(INTDIR)\pngrio.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libpng\pngrtran.c

"$(INTDIR)\pngrtran.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libpng\pngrutil.c

"$(INTDIR)\pngrutil.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libpng\pngset.c

"$(INTDIR)\pngset.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libpng\pngtrans.c

"$(INTDIR)\pngtrans.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libpng\pngwio.c

"$(INTDIR)\pngwio.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libpng\pngwrite.c

"$(INTDIR)\pngwrite.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libpng\pngwtran.c

"$(INTDIR)\pngwtran.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libpng\pngwutil.c

"$(INTDIR)\pngwutil.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jcapimin.c

"$(INTDIR)\jcapimin.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jcapistd.c

"$(INTDIR)\jcapistd.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jccoefct.c

"$(INTDIR)\jccoefct.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jccolor.c

"$(INTDIR)\jccolor.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jcdctmgr.c

"$(INTDIR)\jcdctmgr.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jchuff.c

"$(INTDIR)\jchuff.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jcinit.c

"$(INTDIR)\jcinit.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jcmainct.c

"$(INTDIR)\jcmainct.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jcmarker.c

"$(INTDIR)\jcmarker.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jcmaster.c

"$(INTDIR)\jcmaster.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jcomapi.c

"$(INTDIR)\jcomapi.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jcparam.c

"$(INTDIR)\jcparam.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jcphuff.c

"$(INTDIR)\jcphuff.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jcprepct.c

"$(INTDIR)\jcprepct.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jcsample.c

"$(INTDIR)\jcsample.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jctrans.c

"$(INTDIR)\jctrans.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jdapimin.c

"$(INTDIR)\jdapimin.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jdapistd.c

"$(INTDIR)\jdapistd.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jdatadst.c

"$(INTDIR)\jdatadst.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jdatasrc.c

"$(INTDIR)\jdatasrc.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jdcoefct.c

"$(INTDIR)\jdcoefct.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jdcolor.c

"$(INTDIR)\jdcolor.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\transupp.c

"$(INTDIR)\transupp.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jaricom.c

"$(INTDIR)\jaricom.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jdarith.c

"$(INTDIR)\jdarith.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jcarith.c

"$(INTDIR)\jcarith.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jddctmgr.c

"$(INTDIR)\jddctmgr.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jdhuff.c

"$(INTDIR)\jdhuff.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jdinput.c

"$(INTDIR)\jdinput.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jdmainct.c

"$(INTDIR)\jdmainct.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jdmarker.c

"$(INTDIR)\jdmarker.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jdmaster.c

"$(INTDIR)\jdmaster.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jdmerge.c

"$(INTDIR)\jdmerge.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jdpostct.c

"$(INTDIR)\jdpostct.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jdsample.c

"$(INTDIR)\jdsample.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jdtrans.c

"$(INTDIR)\jdtrans.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jerror.c

"$(INTDIR)\jerror.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jfdctflt.c

"$(INTDIR)\jfdctflt.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jfdctfst.c

"$(INTDIR)\jfdctfst.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jfdctint.c

"$(INTDIR)\jfdctint.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jidctflt.c

"$(INTDIR)\jidctflt.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jidctfst.c

"$(INTDIR)\jidctfst.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jidctint.c

"$(INTDIR)\jidctint.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jmemmgr.c

"$(INTDIR)\jmemmgr.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jmemnobs.c

"$(INTDIR)\jmemnobs.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jquant1.c

"$(INTDIR)\jquant1.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jquant2.c

"$(INTDIR)\jquant2.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libjpeg\jutils.c

"$(INTDIR)\jutils.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libungif\dgif_lib.c

"$(INTDIR)\dgif_lib.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libungif\egif_lib.c

"$(INTDIR)\egif_lib.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libungif\gif_err.c

"$(INTDIR)\gif_err.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libungif\gifalloc.c

"$(INTDIR)\gifalloc.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\libungif\gif_hash.c

"$(INTDIR)\gif_hash.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\afterbase.c

"$(INTDIR)\afterbase.obj" : $(SOURCE) "$(INTDIR)"


SOURCE=.\ascmap.c

"$(INTDIR)\ascmap.obj" : $(SOURCE) "$(INTDIR)"


SOURCE=.\asfont.c

"$(INTDIR)\asfont.obj" : $(SOURCE) "$(INTDIR)"


SOURCE=.\asimage.c

"$(INTDIR)\asimage.obj" : $(SOURCE) "$(INTDIR)"

SOURCE=.\asstorage.c

"$(INTDIR)\asstorage.obj" : $(SOURCE) "$(INTDIR)"

SOURCE=.\asimagexml.c

"$(INTDIR)\asimagexml.obj" : $(SOURCE) "$(INTDIR)"


SOURCE=.\asvisual.c

"$(INTDIR)\asvisual.obj" : $(SOURCE) "$(INTDIR)"


SOURCE=.\blender.c

"$(INTDIR)\blender.obj" : $(SOURCE) "$(INTDIR)"


SOURCE=.\bmp.c

"$(INTDIR)\bmp.obj" : $(SOURCE) "$(INTDIR)"


SOURCE=.\char2uni.c

"$(INTDIR)\char2uni.obj" : $(SOURCE) "$(INTDIR)"


SOURCE=.\export.c

"$(INTDIR)\export.obj" : $(SOURCE) "$(INTDIR)"


SOURCE=.\imencdec.c

"$(INTDIR)\imencdec.obj" : $(SOURCE) "$(INTDIR)"


SOURCE=.\import.c

"$(INTDIR)\import.obj" : $(SOURCE) "$(INTDIR)"


SOURCE=.\pixmap.c

"$(INTDIR)\pixmap.obj" : $(SOURCE) "$(INTDIR)"


SOURCE=.\transform.c

"$(INTDIR)\transform.obj" : $(SOURCE) "$(INTDIR)"


SOURCE=.\ungif.c

"$(INTDIR)\ungif.obj" : $(SOURCE) "$(INTDIR)"


SOURCE=.\xcf.c

"$(INTDIR)\xcf.obj" : $(SOURCE) "$(INTDIR)"


SOURCE=.\ximage.c

"$(INTDIR)\ximage.obj" : $(SOURCE) "$(INTDIR)"


SOURCE=.\xpm.c

"$(INTDIR)\xpm.obj" : $(SOURCE) "$(INTDIR)"

SOURCE=.\draw.c

"$(INTDIR)\draw.obj" : $(SOURCE) "$(INTDIR)"

SOURCE=.\scanline.c

"$(INTDIR)\scanline.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

