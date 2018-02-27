copy_standalone(../../../test/MnSim .
    DemoFumili.cxx
    DemoGaussSim.cxx
    FlatRandomGen.h
    GaussDataGen.cxx
    GaussDataGen.h
    GaussFcn.cxx
    GaussFcn.h
    GaussFcn2.cxx
    GaussFcn2.h
    GaussFunction.h
    GaussRandomGen.h
    GaussianModelFunction.h
    ParallelTest.cxx
    PaulTest.cxx
    PaulTest2.cxx
    PaulTest3.cxx
    PaulTest4.cxx
    ReneTest.cxx
    demoMinimizer.cxx
    paul.txt
    paul2.txt
    paul3.txt
    paul4.txt
)

add_library(GaussSim
    GaussDataGen.cxx
    GaussFcn.cxx
    GaussFcn2.cxx
    )
target_include_directories(GaussSim PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(GaussSim
    PUBLIC
    Minuit2)


add_executable( demoMinimizer
    demoMinimizer.cxx
    )

target_include_directories(demoMinimizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(demoMinimizer PUBLIC Minuit2)

add_executable(DemoFumili
    DemoFumili.cxx
    )

target_include_directories(DemoFumili PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(DemoFumili PUBLIC Minuit2 GaussSim)

add_executable(DemoGaussSim
    DemoGaussSim.cxx
)

target_include_directories(DemoGaussSim PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(DemoGaussSim PUBLIC Minuit2 GaussSim)

add_executable(ParallelTest
    ParallelTest.cxx
    )

target_include_directories(ParallelTest PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ParallelTest PUBLIC Minuit2)

add_executable(PaulTest
    PaulTest.cxx
    )

target_include_directories(PaulTest PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(PaulTest PUBLIC Minuit2 GaussSim)

add_executable(PaulTest2
    PaulTest2.cxx
    )

target_include_directories(PaulTest2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(PaulTest2 PUBLIC Minuit2 GaussSim)

add_executable(PaulTest3
    PaulTest3.cxx
    )

target_include_directories(PaulTest3 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(PaulTest3 PUBLIC Minuit2 GaussSim)

add_executable(PaulTest4
    PaulTest4.cxx
    )

target_include_directories(PaulTest4 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(PaulTest4 PUBLIC Minuit2)

add_executable(ReneTest
    ReneTest.cxx
    )

target_include_directories(ReneTest PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ReneTest PUBLIC Minuit2)

