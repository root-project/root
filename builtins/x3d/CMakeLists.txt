# This ROOT package depends on a hard-coded builtin C library (x3d) that was forked in 1993 and slightly adapted for ROOT. See
# https://github.com/root-project/root/blob/852600061bcacd9b255d44f6312c96b6b1e00a2d/x3d/inc/x3d.h
# https://github.com/root-project/root/blob/852600061bcacd9b255d44f6312c96b6b1e00a2d/x3d/src/x3d.c
# vs http://hpux.connect.org.uk/hppd/hpux/X11/Viewers/x3d-2.2/ download source and rename from .tar.gz to just .tar

add_library(ROOTx3d STATIC)
if(NOT WIN32 AND x11)
  target_sources(ROOTx3d PRIVATE
      x3d.h # private header
      x3d.c
  )
  target_link_libraries(ROOTx3d PRIVATE ${X11_LIBRARIES})
endif()
# Now the ROOT extra additions to the C library:
target_sources(ROOTx3d PRIVATE
    X3DBuffer.c
    X3DDefs.h # private header
  PUBLIC
    inc/X3DBuffer.h # public header
)
set_property(TARGET ROOTx3d PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(ROOTx3d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/inc)
