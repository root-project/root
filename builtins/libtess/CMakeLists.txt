add_library(tessglu STATIC)

# This is a hard-coded copy, almost verbatim, of https://gitlab.freedesktop.org/mesa/glu/-/tree/master/src/libtess
# TODO convert into full builtin by moving this to LCG package and download from link

target_sources(tessglu PRIVATE
  memalloc.c
  dict.c
  geom.c
  mesh.c
  priorityq.c
  normal.c
  render.c
  sweep.c
  tess.c
  tessmono.c
)

set_property(TARGET tessglu PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(tessglu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
