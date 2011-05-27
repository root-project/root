Set(TestSource
    testTMath.cxx
    testBinarySearch.cxx
    testSortOrder.cxx
    stressTMath.cxx
    stressTF1.cxx
    testIntegration.cxx
    testRootFinder.cxx
    kDTreeTest.cxx
   )

Set(TestSourceGraphics
     binarySearchTime.cxx
     stdsort.cxx
     testSpecFuncErf.cxx
     testSpecFuncGamma.cxx
     testSpecFuncBeta.cxx
     testSpecFuncBetaI.cxx
     testIntegrationMultiDim.cxx
)

Set(DEPENDENCIES Core Cint RIO Net Hist Graf Graf3d Gpad Tree 
                 Rint Postscript Matrix Physics MathCore Thread 
                 freetype
   ) 

Set(INCLUDE_DIRECTORIES
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${ROOT_INCLUDE_DIR} 
   ) 
include_directories(${INCLUDE_DIRECTORIES})

ForEach(_file ${TestSource})
  Get_Filename_Component(_testname ${_file} NAME_WE)
  Add_Executable(${_testname} ${_file})
  Target_Link_Libraries(${_testname} ${DEPENDENCIES} GX11 GX11TTF Gui ${CMAKE_THREAD_LIBS_INIT}) 
  Configure_File(${PROJECT_SOURCE_DIR}/cmake/scripts/root_test.sh.in
               ${CMAKE_CURRENT_BINARY_DIR}/${_testname}.sh) 
  Add_Test(mathcore/${_testname} ${CMAKE_CURRENT_BINARY_DIR}/${_testname}.sh)
  Add_Dependencies(check ${_testname})
EndForEach(_file ${TestSource})

ForEach(_file ${TestSourceGraphics})
  Get_Filename_Component(_testname ${_file} NAME_WE)
  Add_Executable(${_testname} ${_file})
  Target_Link_Libraries(${_testname} ${DEPENDENCIES} GX11 GX11TTF Gui ${CMAKE_THREAD_LIBS_INIT}) 
  Configure_File(${PROJECT_SOURCE_DIR}/cmake/scripts/root_test_nographic.sh.in
               ${CMAKE_CURRENT_BINARY_DIR}/${_testname}.sh) 
  Add_Test(mathcore/${_testname} ${CMAKE_CURRENT_BINARY_DIR}/${_testname}.sh)
  Add_Dependencies(check ${_testname})
EndForEach(_file ${TestSourceGraphics})
