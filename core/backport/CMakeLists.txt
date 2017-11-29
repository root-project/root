############################################################################
# CMakeLists.txt file for building ROOT core/backport package
############################################################################

ROOT_GLOB_HEADERS(headers_h ${CMAKE_CURRENT_SOURCE_DIR}/inc/*.h)
ROOT_GLOB_HEADERS(headers_hxx ${CMAKE_CURRENT_SOURCE_DIR}/inc/ROOT/*.hxx)
set(headers ${headers_h} ${headers_hxx})
list(REMOVE_ITEM headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/libcpp_string_view.h)
list(REMOVE_ITEM headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/RWrap_libcpp_string_view.h)

ROOT_INSTALL_HEADERS()

