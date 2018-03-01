
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

copy_standalone(../../../../math/mathcore/inc/Fit ../../inc/Fit ${FIT_HEADERS})
copy_standalone(../../../../math/mathcore/inc/Math ../../inc/Math ${MATH_HEADERS})
copy_standalone(../../../../math/mathcore/src . ${MATH_SOURCES})

prepend_path(MATH_HEADERS ../../inc/Math ${MATH_HEADERS})
prepend_path(FIT_HEADERS ../../inc/Fit ${FIT_HEADERS})

add_library(Math
    ${MATH_SOURCES}
    ${MATH_HEADERS}
    ${FIT_HEADERS}
    )

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

install(TARGETS Math DESTINATION lib EXPORT Minuit2-targets)

target_compile_features(Math PUBLIC cxx_auto_type cxx_static_assert)
set_target_properties(Math PROPERTIES CXX_EXTENSIONS OFF)


install(TARGETS Math
        EXPORT Minuit2Config
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        PUBLIC_HEADER DESTINATION include)
