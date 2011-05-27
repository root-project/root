# TODO: Check if we have to install the buildin version of
# libAfterImage or if we can use the system version of 
# libAfterImage. We have to create a FindAfterImage.cmake 
# script and search for the system version of
# libAfterImage if not set buildin version of libAfterImage.
# Up to now we don't check and install the buildin version anyway.

# This is not a verry clean solution, but the problem is that AfterImage has its
# own tarfile and its own buildsystem. So we have to unpack the tarfile and
# then call the build system of pcre. The created library is imported into
# the scope of cmake, so even make clean works.

if(WIN32)
  set(afterimagelib  ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libAfterImage.lib)
  set(afterimageliba ${CMAKE_CURRENT_BINARY_DIR}/libAfterImage/libAfterImage.lib)
  if(winrtdebug)
    set(astepbld "libAfterImage - Win32 Debug")
  else()
    set(astepbld "libAfterImage - Win32 Release")  
  endif()	

add_custom_command( OUTPUT ${afterimageliba}
                    COMMAND cmake -E copy_directory  ${CMAKE_CURRENT_SOURCE_DIR}/src/libAfterImage libAfterImage
                    COMMAND echo "*** Building ${afterimageliba}"
                    COMMAND cmake -E chdir libAfterImage 
                            nmake -nologo -f libAfterImage.mak FREETYPEDIRI=-I${FREETYPE_INCLUDE_DIR} 
                            CFG=${astepbld} NMAKECXXFLAGS="${CMAKE_CXX_FLAGS} /wd4244")
else() 
  set(afterimagelib  ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libAfterImage.a)
  set(afterimageliba ${CMAKE_CURRENT_BINARY_DIR}/libAfterImage/libAfterImage.a)

  set(AFTER_CC ${CMAKE_C_COMPILER})
  set(AFTER_CFLAGS "-O")
  if(CMAKE_C_COMPILER MATCHES icc)
    set(AFTER_CFLAGS "${AFTER_CFLAGS} -wd188 -wd869 -wd2259 -wd1418 -wd1419 -wd593 -wd981 -wd1599 -wd181 -wd177 -wd1572")
  endif()
  if(ROOT_ARCHITECTURE MATCHES linuxx8664icc)
    set(AFTER_CFLAGS "${AFTER_CFLAGS} -m64 -O")
  elseif(ROOT_ARCHITECTURE MATCHES linuxx8664gcc)
    set(AFTER_CFLAGS "-m64 -O")
    set(AFTER_MMX "--enable-mmx-optimization=no")
  elseif(ROOT_ARCHITECTURE MATCHES linuxicc)
    set(AFTER_CFLAGS "${AFTER_CFLAGS} -m32 -O")
  elseif(ROOT_ARCHITECTURE MATCHES linuxppc64gcc)
    set(AFTER_CFLAGS "-m64 -O")
  elseif(ROOT_ARCHITECTURE MATCHES linux)
    set(AFTER_CFLAGS "-m32 -O")
  elseif(ROOT_ARCHITECTURE MATCHES macosx64)
    set(AFTER_CFLAGS "-m64 -O")
  elseif(ROOT_ARCHITECTURE MATCHES macosx)
    set(AFTER_CFLAGS "-m32 -O")
  elseif(ROOT_ARCHITECTURE MATCHES solarisCC5)
    set(AFTER_CFLAGS "${AFTER_CFLAGS} --erroff=E_WHITE_SPACE_IN_DIRECTIVE")
  elseif(ROOT_ARCHITECTURE MATCHES solaris64CC5)
    set(AFTER_CC "cc")
    set(AFTER_CFLAGS "${AFTER_CFLAGS} -m64 -KPIC --erroff=E_WHITE_SPACE_IN_DIRECTIVE")
  elseif(ROOT_ARCHITECTURE MATCHES sgicc64)
    set(AFTER_CC "gcc")  
    set(AFTER_CFLAGS "-mabi=64 -O")
  elseif(ROOT_ARCHITECTURE MATCHES hpuxia64acc)
    set(AFTER_CC "cc")  
    set(AFTER_CFLAGS "+DD64 -Ae +W863 -O")
    set(AFTER_ALT "gcc -mlp64")
  endif()

  if(JPEG_FOUND)
    set(JPEGINCLUDE "--with-jpeg-includes=-I${JPEG_INCLUDE_DIR}")
  endif()
  if(PNG_FOUND)
    #foreach(_dir ${PNG_INCLUDE_DIR})
    #  set(PNG_INC_DIR "${PNG_INC_DIR} -I${_dir}")
	  #endforeach()
    #set(PNGINCLUDE "--with-png-includes=${PNG_INC_DIR}")
    set(PNGINCLUDE "--with-png-includes=${PNG_PNG_INCLUDE_DIR}")
  endif()
  if(TIFF_FOUND)
    set(TIFFINCLUDE "--with-tiff-includes=-I${TIFF_INCLUDE_DIR}")
  else()
    set(TIFFINCLUDE "--with-tiff=no")
  endif()
  if(GIF_FOUND)
    set(GIFINCLUDE "--with-gif-includes=-I${GIF_INCLUDE_DIR}")
  endif()
  set(TTFINCLUDE "--with-ttf-includes=-I${FREETYPE_INCLUDE_DIR}")

  #---copy files from source directory to build directory------------------------------
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libAfterImage
                     COMMAND cmake -E copy_directory  ${CMAKE_CURRENT_SOURCE_DIR}/src/libAfterImage libAfterImage)

  #---configure and make --------------------------------------------------------------
  add_custom_command(OUTPUT ${afterimageliba}  
                   COMMAND GNUMAKE=make CC=${AFTER_CC} CFLAGS=${AFTER_CFLAGS} ./configure --with-ttf ${TTFINCLUDE} --with-afterbase=no --without-svg --disable-glx ${AFTER_MMX} ${AFTER_DBG} --with-builtin-ungif ${GIFINCLUDE} --with-jpeg ${JPEGINCLUDE} --with-png ${PNGINCLUDE} ${TIFFINCLUDE} > /dev/null 2>& 1
                   COMMAND make > /dev/null 2>& 1
                   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/libAfterImage 
                   DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libAfterImage
                  )
endif()

#---copy the created library into the library directory in the build directory
if(ROOT_PLATFORM MATCHES macosx)
  add_custom_command(OUTPUT ${afterimagelib}
                     COMMAND cmake -E copy_if_different ${afterimageliba} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
                     COMMAND ranlib ${afterimagelib}
                     DEPENDS ${afterimageliba} )
else()
  add_custom_command(OUTPUT ${afterimagelib}
                     COMMAND cmake -E copy_if_different ${afterimageliba} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
                     DEPENDS ${afterimageliba})
endif()
                
# create a target which will always be build and does actually nothing. The target is only
# needed that the dependencies are build, f they are not up to date. If everything is up to
# date nothing is done. This target depends on the libAfterImage.a in the library directory of the 
# build directory.
add_custom_target(AFTERIMAGE DEPENDS ${afterimagelib} )
set_target_properties(AFTERIMAGE PROPERTIES FOLDER Builtins)

install(FILES ${afterimagelib} DESTINATION lib)
