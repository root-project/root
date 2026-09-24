add_library(bvh_c SHARED bvh.cpp)
target_link_libraries(bvh_c PRIVATE bvh)
target_compile_definitions(bvh_c PRIVATE -DBVH_BUILD_API)
target_include_directories(bvh_c INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>)
set_target_properties(bvh_c PROPERTIES
    CXX_STANDARD 20
    CXX_VISIBILITY_PRESET hidden
    INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)

if (BVH_C_API_STATIC_LINK_STDLIB)
    # Link statically against standard C++ library
    target_link_options(bvh_c PRIVATE $<$<C_COMPILER_ID:GNU,Clang>:-static-libstdc++>)
endif()

if (BVH_C_API_UNSAFE_CASTS)
    target_compile_definitions(bvh_c PRIVATE -DBVH_C_API_UNSAFE_CASTS)
endif()
