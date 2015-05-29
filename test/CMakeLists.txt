# CMakeLists.txt for the ROOT test programs.
# This shows nicely how to compile and link applications
# using the ROOT libraries on all supported platforms.
#
# Author: Pere Mato, 25/10/2010
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(test)
find_package(ROOT REQUIRED)

#---Copy the CTestCustom.cmake file into the build directory---------------------------------
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CTestCustom.cmake ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
enable_testing()

if(CMAKE_SYSTEM_NAME MATCHES Darwin)
  # To avoid to set ld_Library_path to locate the test libraries
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,@loader_path/.")
endif()

include_directories(${ROOT_INCLUDE_DIRS})
if(DEFINED ROOT_SOURCE_DIR)  # Testing using the binary tree
  include_directories(${ROOT_SOURCE_DIR}/tutorials)
  include_directories(${ROOT_SOURCE_DIR}/tmva/inc)
  set(ROOT_root_CMD ${ROOT_BINARY_DIR}/root.exe)
else()                       # Testing using an installation (assuming access to ROOT CMake modules)
  include_directories(${ROOT_INCLUDE_DIRS}/../tutorials)
  add_definitions(${ROOT_DEFINITIONS})
  include(RootNewMacros)
endif()

#---environment-------------------------------------------------------------------------------
ROOT_ADD_TEST(show-environment COMMAND ${CMAKE_COMMAND} -E environment)

#---hworld------------------------------------------------------------------------------------
ROOT_EXECUTABLE(hworld hworld.cxx LIBRARIES Gpad)

#---event-------------------------------------------------------------------------------------
ROOT_GENERATE_DICTIONARY(EventDict ${CMAKE_CURRENT_SOURCE_DIR}/Event.h MODULE Event LINKDEF EventLinkDef.h)
ROOT_LINKER_LIBRARY(Event Event.cxx EventDict.cxx LIBRARIES Hist MathCore)

ROOT_EXECUTABLE(eventexe MainEvent.cxx LIBRARIES Event RIO Tree Hist Net)
ROOT_ADD_TEST(test-event COMMAND eventexe)

#---hsimple------------------------------------------------------------------------------------
#ROOT_EXECUTABLE(hsimple hsimple.cxx LIBRARIES RIO Tree Hist)
#ROOT_ADD_TEST(test-hsimple COMMAND hsimple)

#--minexam-------------------------------------------------------------------------------------
ROOT_EXECUTABLE(minexam minexam.cxx LIBRARIES Minuit)
ROOT_ADD_TEST(test-minexam COMMAND minexam)

#--tstring-------------------------------------------------------------------------------------
ROOT_EXECUTABLE(tstring tstring.cxx LIBRARIES Core)
configure_file(tstring.cxx tstring.cxx COPYONLY)
ROOT_ADD_TEST(test-tstring COMMAND tstring)

#--tcollex-------------------------------------------------------------------------------------
ROOT_EXECUTABLE(tcollex tcollex.cxx LIBRARIES Core)
ROOT_ADD_TEST(test-tcollex COMMAND tcollex)

#--tcollbm------------------------------------------------------------------------------------
ROOT_EXECUTABLE(tcollbm tcollbm.cxx LIBRARIES Core MathCore)
ROOT_ADD_TEST(test-tcollbm COMMAND tcollbm 1000 100000)

#--vvector------------------------------------------------------------------------------------
ROOT_EXECUTABLE(vvector vvector.cxx LIBRARIES Core Matrix RIO)
ROOT_ADD_TEST(test-vvector COMMAND vvector)

#--vmatrix------------------------------------------------------------------------------------
ROOT_EXECUTABLE(vmatrix vmatrix.cxx LIBRARIES Core Matrix RIO)
ROOT_ADD_TEST(test-vmatrix COMMAND vmatrix)

#--vlazy------------------------------------------------------------------------------------
ROOT_EXECUTABLE(vlazy vlazy.cxx LIBRARIES Core Matrix)
ROOT_ADD_TEST(test-vlazy COMMAND vlazy)

#--helloso------------------------------------------------------------------------------------
ROOT_GENERATE_DICTIONARY(HelloDict ${CMAKE_CURRENT_SOURCE_DIR}/Hello.h MODULE Hello)
ROOT_LINKER_LIBRARY(Hello Hello.cxx HelloDict.cxx LIBRARIES Graf Gpad)

#--Aclockso------------------------------------------------------------------------------------
ROOT_GENERATE_DICTIONARY(AclockDict ${CMAKE_CURRENT_SOURCE_DIR}/Aclock.h MODULE Aclock)
ROOT_LINKER_LIBRARY(Aclock Aclock.cxx AclockDict.cxx LIBRARIES Graf Gpad)

#--bench------------------------------------------------------------------------------------
ROOT_GENERATE_DICTIONARY(TBenchDict ${CMAKE_CURRENT_SOURCE_DIR}/TBench.h MODULE TBench LINKDEF benchLinkDef.h)
ROOT_LINKER_LIBRARY(TBench TBench.cxx TBenchDict.cxx LIBRARIES Core MathCore RIO Tree)
ROOT_EXECUTABLE(bench bench.cxx LIBRARIES Core TBench)
ROOT_ADD_TEST(test-bench COMMAND bench)

#--stress------------------------------------------------------------------------------------
ROOT_EXECUTABLE(stress stress.cxx LIBRARIES Event Core Hist RIO Tree Gpad Postscript)
ROOT_ADD_TEST(test-stress COMMAND stress -b FAILREGEX "FAILED|Error in"
                          DEPENDS test-event)

#--stressShapes------------------------------------------------------------------------------------
ROOT_EXECUTABLE(stressShapes stressShapes.cxx LIBRARIES  Geom Tree GenVector Gpad)
ROOT_ADD_TEST(test-stressshapes COMMAND stressShapes -b FAILREGEX "FAILED|Error in")
ROOT_ADD_TEST(test-stressshapes-interpreted COMMAND ${ROOT_root_CMD} -b -q -l ${CMAKE_CURRENT_SOURCE_DIR}/stressShapes.cxx
              FAILREGEX "FAILED|Error in"
)

#--stressGeometry------------------------------------------------------------------------------------
ROOT_EXECUTABLE(stressGeometry stressGeometry.cxx LIBRARIES Geom Tree GenVector Gpad)
ROOT_ADD_TEST(test-stressgeometry COMMAND stressGeometry -b FAILREGEX "FAILED|Error in")
ROOT_ADD_TEST(test-stressgeometry-interpreted COMMAND ${ROOT_root_CMD} -b -q -l ${CMAKE_CURRENT_SOURCE_DIR}/stressGeometry.cxx
              FAILREGEX "FAILED|Error in" DEPENDS test-stressgeometry)

#--stressLinear------------------------------------------------------------------------------------
ROOT_EXECUTABLE(stressLinear stressLinear.cxx LIBRARIES Matrix Hist RIO)
ROOT_ADD_TEST(test-stresslinear COMMAND stressLinear FAILREGEX "FAILED|Error in")
ROOT_ADD_TEST(test-stresslinear-interpreted COMMAND ${ROOT_root_CMD} -b -q -l ${CMAKE_CURRENT_SOURCE_DIR}/stressLinear.cxx
              FAILREGEX "FAILED|Error in" DEPENDS test-stresslinear)

#--stressGraphics------------------------------------------------------------------------------------
ROOT_EXECUTABLE(stressGraphics stressGraphics.cxx LIBRARIES Graf Gpad Postscript)
configure_file(stressGraphics.ref stressGraphics.ref COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../tutorials/graphics/earth.dat earth.dat COPYONLY)
ROOT_ADD_TEST(test-stressgraphics ENVIRONMENT LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH} COMMAND stressGraphics -b -k FAILREGEX "FAILED|Error in")
ROOT_ADD_TEST(test-stressgraphics-interpreted COMMAND ${ROOT_root_CMD} -b -q -l ${CMAKE_CURRENT_SOURCE_DIR}/stressGraphics.cxx
              FAILREGEX "FAILED|Error in" DEPENDS test-stressgraphics)

#--stressHistogram------------------------------------------------------------------------------------
ROOT_EXECUTABLE(stressHistogram stressHistogram.cxx LIBRARIES Hist RIO)
ROOT_ADD_TEST(test-stresshistogram COMMAND stressHistogram FAILREGEX "FAILED|Error in")
ROOT_ADD_TEST(test-stresshistogram-interpreted COMMAND ${ROOT_root_CMD} -b -q -l ${CMAKE_CURRENT_SOURCE_DIR}/stressHistogram.cxx
              FAILREGEX "FAILED|Error in" DEPENDS test-stresshistogram)

#--stressGUI---------------------------------------------------------------------------------------
ROOT_EXECUTABLE(stressGUI stressGUI.cxx LIBRARIES Gui Recorder GuiHtml ASImageGui)
#---Cannot run GUI test in batch mode--------------------
#configure_file(stressGUI.ref stressGUI.ref COPYONLY)
#ROOT_ADD_TEST(test-stressgui COMMAND stressGUI -ref FAILREGEX "FAILED|Error in")

#--stressSpectrum----------------------------------------------------------------------------------
ROOT_EXECUTABLE(stressSpectrum stressSpectrum.cxx LIBRARIES Hist Spectrum Gpad)
ROOT_ADD_TEST(test-stressspectrum COMMAND stressSpectrum -b FAILREGEX "FAILED|Error in")
ROOT_ADD_TEST(test-stressspectrum-interpreted COMMAND ${ROOT_root_CMD} -b -q -l ${CMAKE_CURRENT_SOURCE_DIR}/stressSpectrum.cxx
              FAILREGEX "FAILED|Error in" DEPENDS test-stressspectrum)

#--stressVector------------------------------------------------------------------------------------
ROOT_EXECUTABLE(stressVector stressVector.cxx LIBRARIES Physics GenVector)
ROOT_ADD_TEST(test-stressvector COMMAND stressVector FAILREGEX "FAILED|Error in")
ROOT_ADD_TEST(test-stressvector-interpreted COMMAND ${ROOT_root_CMD} -b -q -l ${CMAKE_CURRENT_SOURCE_DIR}/stressVector.cxx
              FAILREGEX "FAILED|Error in" DEPENDS test-stressvector)

#--stressTMVA--------------------------------------------------------------------------------------
if(ROOT_tmva_FOUND)
  ROOT_EXECUTABLE(stressTMVA stressTMVA.cxx LIBRARIES TMVA)
  ROOT_ADD_TEST(test-stresstmva COMMAND stressTMVA -b)
  ROOT_ADD_TEST(test-stresstmva-interpreted COMMAND ${ROOT_root_CMD} -b -q -l ${CMAKE_CURRENT_SOURCE_DIR}/stressTMVA.cxx
                FAILREGEX "FAILED|Error in" DEPENDS test-stresstmva)
endif()

#--stressMathMore----------------------------------------------------------------------------------
if(ROOT_mathmore_FOUND)
  ROOT_EXECUTABLE(stressMathMore stressMathMore.cxx LIBRARIES MathMore)
  ROOT_ADD_TEST(test-stressmathmore COMMAND stressMathMore FAILREGEX "FAILED|Error in")
  ROOT_ADD_TEST(test-stressmathmore-interpreted COMMAND ${ROOT_root_CMD} -b -q -l ${CMAKE_CURRENT_SOURCE_DIR}/stressMathMore.cxx
                FAILREGEX "FAILED|Error in" DEPENDS test-stressmathmore)
endif()

#--stressMathCore----------------------------------------------------------------------------------
ROOT_GENERATE_DICTIONARY(TrackMathCoreDictionary ${CMAKE_CURRENT_SOURCE_DIR}/TrackMathCore.h MODULE TrackMathCoreDict LINKDEF TrackMathCoreLinkDef.h)
ROOT_LINKER_LIBRARY(TrackMathCoreDict TrackMathCoreDictionary.cxx LIBRARIES Core MathCore RIO GenVector)
ROOT_EXECUTABLE(stressMathCore stressMathCore.cxx LIBRARIES MathCore Hist RIO Tree GenVector)
ROOT_ADD_TEST(test-stressmathcore COMMAND stressMathCore FAILREGEX "FAILED|Error in")
ROOT_ADD_TEST(test-stressmathcore-interpreted COMMAND ${ROOT_root_CMD} -b -q -l ${CMAKE_CURRENT_SOURCE_DIR}/stressMathCore.cxx
              FAILREGEX "FAILED|Error in" DEPENDS test-stressmathcore)

#--stressRooFit----------------------------------------------------------------------------------
if(ROOT_roofit_FOUND)
  ROOT_EXECUTABLE(stressRooFit stressRooFit.cxx LIBRARIES RooFit)
  ROOT_ADD_TEST(test-stressroofit COMMAND stressRooFit FAILREGEX "FAILED|Error in")
  ROOT_ADD_TEST(test-stressroofit-interpreted COMMAND ${ROOT_root_CMD} -b -q -l ${CMAKE_CURRENT_SOURCE_DIR}/stressRooFit.cxx
                FAILREGEX "FAILED|Error in" DEPENDS test-stressroofit)
endif()

#--stressRooStats----------------------------------------------------------------------------------
if(ROOT_roofit_FOUND)
  ROOT_EXECUTABLE(stressRooStats stressRooStats.cxx LIBRARIES RooStats)
  configure_file(stressRooStats_ref.root stressRooStats_ref.root COPYONLY)
  ROOT_ADD_TEST(test-stressroostats COMMAND stressRooStats FAILREGEX "FAILED|Error in")
  ROOT_ADD_TEST(test-stressroostats-interpreted COMMAND ${ROOT_root_CMD} -b -q -l ${CMAKE_CURRENT_SOURCE_DIR}/stressRooStats.cxx
                FAILREGEX "FAILED|Error in" DEPENDS test-stressroostats)
endif()

#--stressHistFactory--------------------------------------------------------------------------------
if(ROOT_roofit_FOUND)
if(ROOT_xml_FOUND)
  ROOT_EXECUTABLE(stressHistFactory stressHistFactory.cxx LIBRARIES RooStats HistFactory XMLParser)
  configure_file(HistFactoryTest.tar HistFactoryTest.tar COPYONLY)
  configure_file(stressHistFactory_ref.root stressHistFactory_ref.root COPYONLY)
  ROOT_ADD_TEST(test-stressHistFactory ENVIRONMENT ROOTSYS=${CMAKE_BINARY_DIR} COMMAND stressHistFactory FAILREGEX "FAILED|Error in")
  ROOT_ADD_TEST(test-stressHistFactory-interpreted COMMAND ${ROOT_root_CMD} -b -q -l ${CMAKE_CURRENT_SOURCE_DIR}/stressHistFactory.cxx FAILREGEX "FAILED|Error in" DEPENDS test-stressHistFactory)
endif()
endif()

#--stressFit---------------------------------------------------------------------------------
ROOT_EXECUTABLE(stressFit stressFit.cxx LIBRARIES MathCore Matrix)
ROOT_ADD_TEST(test-stressfit COMMAND stressFit FAILREGEX "FAILED|Error in")
ROOT_ADD_TEST(test-stressfit-interpreted COMMAND ${ROOT_root_CMD} -b -q -l ${CMAKE_CURRENT_SOURCE_DIR}/stressFit.cxx
              FAILREGEX "FAILED|Error in" DEPENDS test-stressfit)

#--stressHistoFit-----------------------------------------------------------------------------
if(ROOT_unuran_FOUND)
  ROOT_EXECUTABLE(stressHistoFit stressHistoFit.cxx LIBRARIES MathCore Matrix Unuran Tree Gpad)
  ROOT_ADD_TEST(test-stresshistofit COMMAND stressHistoFit FAILREGEX "FAILED|Error in")
  ROOT_ADD_TEST(test-stresshistofit-interpreted COMMAND ${ROOT_root_CMD} -b -q -l ${CMAKE_CURRENT_SOURCE_DIR}/stressHistoFit.cxx
                FAILREGEX "FAILED|Error in" DEPENDS test-stresshistofit)
endif()

#--stressEntryList---------------------------------------------------------------------------
ROOT_EXECUTABLE(stressEntryList stressEntryList.cxx LIBRARIES MathCore Tree Hist)
ROOT_ADD_TEST(test-stressentrylist COMMAND stressEntryList -b FAILREGEX "FAILED|Error in")
ROOT_ADD_TEST(test-stressentrylist-interpreted COMMAND ${ROOT_root_CMD} -b -q -l ${CMAKE_CURRENT_SOURCE_DIR}/stressEntryList.cxx
              FAILREGEX "FAILED|Error in" DEPENDS test-stressentrylist)

#--stressIterators---------------------------------------------------------------------------
ROOT_EXECUTABLE(stressIterators stressIterators.cxx LIBRARIES Core)
ROOT_ADD_TEST(test-stressiterators COMMAND stressIterators FAILREGEX "FAILED|Error in")
ROOT_ADD_TEST(test-stressiterators-interpreted COMMAND ${ROOT_root_CMD} -b -q -l ${CMAKE_CURRENT_SOURCE_DIR}/stressIterators.cxx
              FAILREGEX "FAILED|Error in" DEPENDS test-stressiterators)

#--stressInterpreter-------------------------------------------------------------------------
ROOT_EXECUTABLE(stressInterpreter stressInterpreter.cxx LIBRARIES Core)
if(WIN32)
  set(cleantmpfiles POSTCMD cmd /c del AutoDict_*)
endif()
ROOT_ADD_TEST(test-stressinterpreter COMMAND stressInterpreter
              PRECMD ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/stressInterpreter.cxx
                                              $<TARGET_FILE_DIR:stressInterpreter>/stressInterpreter.cxx
              ${cleantmpfiles}
              FAILREGEX Error|FAILED)

#--stressHepix-------------------------------------------------------------------------------
ROOT_EXECUTABLE(stressHepix stressHepix.cxx LIBRARIES Core)
#ROOT_ADD_TEST(test-stressHepix COMMAND stressHepix FAILREGEX "FAILED|Error in")

#--stressProof-------------------------------------------------------------------------------
if(NOT WIN32 AND ROOT_xrootd_FOUND)
  add_custom_target(TestData ALL COMMAND ${CMAKE_COMMAND} -DDST=${CMAKE_SOURCE_DIR}/files -P ${CMAKE_CURRENT_SOURCE_DIR}/rootDownloadData.cmake)
  ROOT_EXECUTABLE(stressProof stressProof.cxx LIBRARIES Proof ProofPlayer Hist)
  ROOT_ADD_TEST(test-stressproof COMMAND stressProof lite://
                                         -h1 ${CMAKE_SOURCE_DIR}/files/h1 -event ${CMAKE_SOURCE_DIR}/files/event
                                         -l /tmp/stressProof-%d.log -cleanlog -catlog -noprogress
                                         ENVIRONMENT PATH=${CMAKE_BINARY_DIR}/bin:$ENV{PATH}
                                         FAILREGEX "FAILED|Error in")
endif()

#--testbits----------------------------------------------------------------------------------
ROOT_EXECUTABLE(testbits testbits.cxx LIBRARIES Core)
ROOT_ADD_TEST(test-testbits COMMAND testbits)

#--ctorture----------------------------------------------------------------------------------
ROOT_EXECUTABLE(ctorture ctorture.cxx LIBRARIES MathCore)
ROOT_ADD_TEST(test-ctorture COMMAND ctorture)

#--qprandom----------------------------------------------------------------------------------
ROOT_EXECUTABLE(QpRandomDriver QpRandomDriver.cxx LIBRARIES Quadp)
ROOT_ADD_TEST(test-qprandom COMMAND QpRandomDriver 10 2 2)

#--threads----------------------------------------------------------------------------------
ROOT_EXECUTABLE(threads threads.cxx LIBRARIES Thread Hist Gpad)
#ROOT_ADD_TEST(test-threads COMMAND threads)

#--stressIOPlugins--------------------------------------------------------------------------
ROOT_EXECUTABLE(stressIOPlugins stressIOPlugins.cxx LIBRARIES Event Core Hist RIO Tree Gpad Postscript)
if(ROOT_xrootd_FOUND)
  ROOT_ADD_TEST(test-stressIOPlugins-xroot COMMAND stressIOPlugins root FAILREGEX "FAILED|Error in")
endif()
if(ROOT_davix_FOUND)
  ROOT_ADD_TEST(test-stressIOPlugins-http COMMAND stressIOPlugins http FAILREGEX "FAILED|Error in")
endif()

#--TestTformula------------------------------------------------------------------------------------
ROOT_EXECUTABLE(TFormulaTests TFormulaTests.cxx LIBRARIES Hist RIO)
ROOT_ADD_TEST(test-TFormulaTests COMMAND TFormulaTests FAILREGEX "FAILED|Error in")

#--TBB basic test----------------------------------------------------------------------------------
if(ROOT_tbb_FOUND)
  include_directories(${TBB_INCLUDE_DIRS})
  ROOT_EXECUTABLE(testTBB testTBB.cxx LIBRARIES ${TBB_LIBRARIES})
  ROOT_ADD_TEST(test-TBB COMMAND testTBB FAILREGEX "FAILED|Error in")
  if(builtin_tbb)
    add_dependencies(testTBB TBB)
  endif()
endif()
