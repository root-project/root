project(minuit2-tests)
find_package(ROOT REQUIRED)

include_directories(${ROOT_INCLUDE_DIRS})

set(TestSource
      testNewMinimizer.cxx 
)

set(TestSourceMnTutorial
    MnTutorial/Quad1FMain.cxx
    MnTutorial/Quad4FMain.cxx
    MnTutorial/Quad8FMain.cxx
    MnTutorial/Quad12FMain.cxx
)

set(TestSourceMnSim 
    MnSim/DemoGaussSim.cxx
    MnSim/DemoFumili.cxx
    MnSim/PaulTest.cxx
    MnSim/PaulTest2.cxx
    MnSim/PaulTest3.cxx
    MnSim/PaulTest4.cxx
    MnSim/ReneTest.cxx
    MnSim/ParallelTest.cxx
    MnSim/demoMinimizer.cxx
)




#---For the simple Minuit2 tests build and defined them---------------
foreach(file ${TestSourceMnTutorial})
  get_filename_component(testname ${file} NAME_WE)
  ROOT_EXECUTABLE(${testname} ${file} LIBRARIES Minuit2)
  ROOT_ADD_TEST(minuit2-${testname} COMMAND ${testname})
endforeach()


ROOT_LINKER_LIBRARY(Minuit2TestMnSim MnSim/GaussDataGen.cxx MnSim/GaussFcn.cxx MnSim/GaussFcn2.cxx LIBRARIES Minuit2)

#input text files
configure_file(MnSim/paul.txt paul.txt COPYONLY)
configure_file(MnSim/paul2.txt paul2.txt COPYONLY)
configure_file(MnSim/paul3.txt paul3.txt COPYONLY)
configure_file(MnSim/paul4.txt paul4.txt COPYONLY)

foreach(file ${TestSourceMnSim})
  get_filename_component(testname ${file} NAME_WE)
  ROOT_EXECUTABLE(${testname} ${file} LIBRARIES Minuit2 Minuit2TestMnSim)
  ROOT_ADD_TEST(minuit2-${testname} COMMAND ${testname}) 
endforeach()

#for the global tests using ROOT libs (Minuit2 should be taken via the PluginManager)

set(RootLibraries Core RIO Net Hist Graf Graf3d Gpad Tree
              Rint Postscript Matrix Physics MathCore Thread)

foreach(file ${TestSource})
  get_filename_component(testname ${file} NAME_WE)
  ROOT_EXECUTABLE(${testname} ${file} LIBRARIES ${RootLibraries} )
  ROOT_ADD_TEST(minuit2-${testname} COMMAND ${testname}) 
endforeach()


