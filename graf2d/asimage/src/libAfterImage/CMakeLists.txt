# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

project(AfterImage)

set(LIB_NAME libAfterImage)

set(H_FILES
    afterbase.h
    afterrootpngwrite.h
    asfont.h
    asimage.h
    asvisual.h
    bmp.h
    colornames.h
    export.h
    import.h
    transform.h
    xcf.h
    xpm.h
    afterimage.h
    ascmap.h
    asim_afterbase.h
    asstorage.h
    blender.h
    char2uni.h
    draw.h
    imencdec.h
    scanline.h
    ungif.h
    ximage.h
    xwrap.h)

set(SRC_FILES
    afterbase.c
    ascmap.c
    asfont.c
    asimage.c
    asstorage.c
    asvisual.c
    blender.c
    bmp.c
    char2uni.c
    export.c
    import.c
    transform.c
    ungif.c
    xcf.c
    ximage.c
    xpm.c
    draw.c
    imencdec.c
    scanline.c
    afterrootpngwrite.c)

add_library(${LIB_NAME} STATIC ${H_FILES} ${SRC_FILES})
set_property(TARGET ${LIB_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

# ASEXTRA_LIBRARIES: PNG::PNG JPEG::JPEG GIF::GIF

target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${LIB_NAME} PRIVATE ${ASEXTRA_LIBRARIES} Freetype::Freetype ZLIB::ZLIB)
if(TARGET X11::X11)
  target_link_libraries(${LIB_NAME} PRIVATE X11::X11)
endif()

include(CheckIncludeFiles)

target_compile_definitions(
  ${LIB_NAME}
  PRIVATE HAVE_ERRNO_H
          HAVE_FREETYPE
          HAVE_FREETYPE_FREETYPE
          HAVE_FT2BUILD_H
          HAVE_INTTYPES_H
          HAVE_MEMORY_H
          HAVE_PROTOTYPES
          HAVE_STDARG_H
          HAVE_STDDEF_H
          HAVE_STRING_H
          HAVE_SYS_STAT_H
          HAVE_SYS_TYPES_H
          HAVE_SYS_WAIT_H
          HAVE_XPM
          HAVE_ZLIB_H
          NO_DEBUG_OUTPUT)

check_include_files(malloc.h PROJECT_HAVE_MALLOC)
if (PROJECT_HAVE_MALLOC)
  target_compile_definitions(
    ${LIB_NAME}
    PRIVATE HAVE_MALLOC_H)
endif()

if(TARGET JPEG::JPEG)
  target_compile_definitions(${LIB_NAME} PRIVATE HAVE_JPEG)
endif()
if(TARGET PNG::PNG)
  target_compile_definitions(${LIB_NAME} PRIVATE HAVE_PNG)
endif()
if(TARGET GIF::GIF)
  target_compile_definitions(${LIB_NAME} PRIVATE HAVE_GIF) #
endif()
if(TARGET TIFF::TIFF)
  target_compile_definitions(${LIB_NAME} PRIVATE HAVE_TIFF) # asimage_tiff
endif()

if(WIN32)
  # Microsoft Visual Studio:
  if(MSVC)
    # Define
    target_compile_definitions(${LIB_NAME} PRIVATE _CRT_SECURE_NO_DEPRECATE
                                                   _MBCS _LIB)
    target_compile_options(${LIB_NAME} PRIVATE /wd4996 /wd4267 /wd4018 /wd4244)

  endif()
  target_compile_definitions(
    ${LIB_NAME}
    PRIVATE NO_DOUBLE_FCLOSE_AFTER_FDOPEN X_DISPLAY_MISSING
            HAVE_UNSIGNED_CHAR HAVE_UNSIGNED_SHORT HAVE_BOOLEAN)
else()
  target_compile_definitions(
    ${LIB_NAME}
    PRIVATE HAVE_SYS_TIME_H
            HAVE_DIRENT_H
            HAVE_LONG_LONG
            HAVE_STDINT_H
            HAVE_STDLIB_H
            HAVE_UNISTD_H
            HAVE_STRINGS_H
            SHAPE
            STDC_HEADERS
            TIME_WITH_SYS_TIME)
  check_include_files(mmintrin.h PROJECT_HAVE_SSE2_INTRINSICS)
  if (PROJECT_HAVE_SSE2_INTRINSICS)
    target_compile_definitions(
      ${LIB_NAME}
      PRIVATE HAVE_MMX)
  endif()
  if(NOT TARGET X11::X11)
    target_compile_definitions(
      ${LIB_NAME} PRIVATE X_DISPLAY_MISSING)
  endif()

endif()
