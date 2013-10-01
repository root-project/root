if(QT4_FOUND AND QT4_USABLE)
   include(${QT_USE_FILE})
   include_directories(${CMAKE_CURRENT_BINARY_DIR})

   add_executable(buddhabrot_sse main.cpp)
   add_target_property(buddhabrot_sse COMPILE_FLAGS "-DVC_IMPL=SSE")
   target_link_libraries(buddhabrot_sse ${QT_LIBRARIES} Vc)

   add_executable(buddhabrot_scalar main.cpp)
   add_target_property(buddhabrot_scalar COMPILE_FLAGS "-DVC_IMPL=Scalar")
   target_link_libraries(buddhabrot_scalar ${QT_LIBRARIES} Vc)

   add_executable(buddhabrot_scalar2 main.cpp)
   add_target_property(buddhabrot_scalar2 COMPILE_FLAGS "-DScalar")
   target_link_libraries(buddhabrot_scalar2 ${QT_LIBRARIES})
endif()
