############################################################################
# CMakeLists.txt file for building ROOT core/clib package
############################################################################

set(sources attach.c
   detach.c
   keys.c
   mfree.c
   mmapsup.c
   mmemalign.c
   mmtrace.c
   mvalloc.c
   snprintf.c
   strlcpy.c
   Demangle.c
   getpagesize.c
   mcalloc.c
   mmalloc.c
   mmcheck.c
   mmstats.c
   mrealloc.c
   sbrksup.c
   strlcat.c
)

set(Clib_dict_headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/strlcpy.h
                      ${CMAKE_CURRENT_SOURCE_DIR}/inc/snprintf.h PARENT_SCOPE)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/res)

ROOT_OBJECT_LIBRARY(Clib ${sources})

ROOT_INSTALL_HEADERS()



