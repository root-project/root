
set(FIT_HEADERS
    ParameterSettings.h
    )

set(MATH_HEADERS
    Error.h
    FitMethodFunction.h
    Functor.h
    GenAlgoOptions.h
    IFunction.h
    IFunctionfwd.h
    IOptions.h
    Minimizer.h
    MinimizerOptions.h
    Util.h
    WrappedFunction.h
    WrappedParamFunction.h
    )

set(MATH_SOURCES
    GenAlgoOptions.cxx
    MinimizerOptions.cxx
    )

copy_standalone(SOURCE ../../../../math/mathcore/inc/Fit DESTINATION ../../inc/Fit
    OUTPUT FIT_HEADERS
    FILES ${FIT_HEADERS})
copy_standalone(SOURCE ../../../../math/mathcore/inc/Math DESTINATION ../../inc/Math
    OUTPUT  MATH_HEADERS
    FILES ${MATH_HEADERS})
copy_standalone(SOURCE ../../../../math/mathcore/src DESTINATION .
    OUTPUT MATH_SOURCES
    FILES ${MATH_SOURCES})

# Adding the headers helps IDEs show the correct headers on targets
add_library(Math
    ${MATH_SOURCES}
    ${MATH_HEADERS}
    ${FIT_HEADERS}
    )

# Add alias for direct inclusion
add_library(Minuit2::Math ALIAS Math)

# Build and install directories are different, using CMake Generator expression
target_include_directories(
    Math
    PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/inc>
    $<INSTALL_INTERFACE:include/Minuit2>
    )

target_compile_definitions(
    Math
    PRIVATE
    MATH_NO_PLUGIN_MANAGER
    )

target_compile_definitions(
    Math
    PUBLIC
    ROOT_Math_VecTypes
    )

target_link_libraries(Math PUBLIC Common)

target_compile_features(Math PUBLIC cxx_auto_type cxx_static_assert)
set_target_properties(Math PROPERTIES CXX_EXTENSIONS OFF)

install(TARGETS Math
        EXPORT Minuit2Targets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        )

install(FILES ${FIT_HEADERS} DESTINATION include/Minuit2/Fit)
install(FILES ${MATH_HEADERS} DESTINATION include/Minuit2/Math)
