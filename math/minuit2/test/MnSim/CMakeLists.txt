
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

