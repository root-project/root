add_executable( Quad1F
    Quad1F.h
    Quad1FMain.cxx
)

target_include_directories(Quad1F PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(Quad1F PUBLIC Minuit2)

add_executable( Quad4F
    Quad4F.h
    Quad4FMain.cxx
)

target_include_directories(Quad4F PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(Quad4F PUBLIC Minuit2)

add_executable( Quad8F
    Quad8F.h
    Quad8FMain.cxx
)

target_include_directories(Quad8F PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(Quad8F PUBLIC Minuit2)

add_executable( Quad12F
    Quad12F.h
    Quad12FMain.cxx
)

target_include_directories(Quad12F PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(Quad12F PUBLIC Minuit2)

