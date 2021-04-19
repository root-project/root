# libAferImage CMakeLists.txt

PROJECT(AFTERIMAGE)
if(MSVC)
  # required for the following feature & bug fix:
  # 3.15: Added $<REMOVE_DUPLICATES:list> generator expression
  # 3.16: Bug fix with CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS: the auto-generated exports
  #       are now updated only when the object files providing the symbols are updated
  cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
else()
  cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
endif()

SET(LIB_NAME libAfterImage)

# Microsoft Visual Studio:
IF(MSVC)
  # Define
  ADD_DEFINITIONS(-D_CRT_SECURE_NO_DEPRECATE)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /DNO_DEBUG_OUTPUT /D_MBCS /D_LIB /wd4996 /wd4267 /wd4018 /wd4244")
ENDIF()

set(FREETYPE_INCLUDE_DIR "" CACHE PATH "Path to Freetype include dir")

if(NOT EXISTS "${FREETYPE_INCLUDE_DIR}/ft2build.h")
  message(ERROR "Can't find ft2build.h in ${FREETYPE_INCLUDE_DIR}")
endif()

INCLUDE_DIRECTORIES(${FREETYPE_INCLUDE_DIR})

set (LIB_DESTINATION "${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX}")

FILE(GLOB H_FILES "*.h")

SET(SRC_FILES
    libpng/png.c libpng/pngerror.c libpng/pngget.c libpng/pngmem.c libpng/pngpread.c libpng/pngread.c
    libpng/pngrio.c libpng/pngrtran.c libpng/pngrutil.c libpng/pngset.c libpng/pngtrans.c libpng/pngwio.c
    libpng/pngwrite.c libpng/pngwtran.c libpng/pngwutil.c
    libjpeg/jcapimin.c libjpeg/jcapistd.c libjpeg/jccoefct.c libjpeg/jccolor.c libjpeg/jcdctmgr.c libjpeg/jchuff.c libjpeg/jcinit.c
    libjpeg/jcmainct.c libjpeg/jcmarker.c libjpeg/jcmaster.c libjpeg/jcomapi.c libjpeg/jcparam.c libjpeg/jcphuff.c libjpeg/jcprepct.c
    libjpeg/jcsample.c libjpeg/jctrans.c libjpeg/jdapimin.c libjpeg/jdapistd.c libjpeg/jdatadst.c libjpeg/jdatasrc.c libjpeg/jdcoefct.c
    libjpeg/jdcolor.c libjpeg/transupp.c libjpeg/jaricom.c libjpeg/jdarith.c libjpeg/jcarith.c libjpeg/jddctmgr.c libjpeg/jdhuff.c
    libjpeg/jdinput.c libjpeg/jdmainct.c libjpeg/jdmarker.c libjpeg/jdmaster.c libjpeg/jdmerge.c libjpeg/jdpostct.c libjpeg/jdsample.c
    libjpeg/jdtrans.c libjpeg/jerror.c libjpeg/jfdctflt.c libjpeg/jfdctfst.c libjpeg/jfdctint.c libjpeg/jidctflt.c libjpeg/jidctfst.c
    libjpeg/jidctint.c libjpeg/jmemmgr.c libjpeg/jmemnobs.c libjpeg/jquant1.c libjpeg/jquant2.c libjpeg/jutils.c
    libungif/dgif_lib.c libungif/egif_lib.c libungif/gif_err.c libungif/gifalloc.c libungif/gif_hash.c afterbase.c ascmap.c asfont.c
    asimage.c asstorage.c asimagexml.c asvisual.c blender.c bmp.c char2uni.c
    export.c import.c pixmap.c transform.c ungif.c xcf.c ximage.c xpm.c draw.c
    imencdec.c scanline.c
)

ADD_LIBRARY(${LIB_NAME} STATIC ${H_FILES} ${SRC_FILES})

install(TARGETS ${LIB_NAME} DESTINATION ${LIB_DESTINATION})

