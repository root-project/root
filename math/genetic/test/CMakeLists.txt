project(genetic-tests)
find_package(ROOT REQUIRED)

include_directories(${ROOT_INCLUDE_DIRS})

set(Libraries Core RIO Net Hist Graf Graf3d Gpad Tree
              Rint Postscript Matrix Physics MathCore Thread Genetic)

set(TestGeneticSource
    testGAMinimizer.cxx
     GAMinTutorial.cxx )




#---Build and add all the defined test in the list---------------
foreach(file ${TestGeneticSource})
  get_filename_component(testname ${file} NAME_WE)
  ROOT_EXECUTABLE(${testname} ${file} LIBRARIES ${Libraries})
  ROOT_ADD_TEST(genetic-${testname} COMMAND ${testname})
endforeach()

