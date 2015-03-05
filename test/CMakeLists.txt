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


if(CMAKE_SYSTEM_NAME MATCHES Darwin)
  # To avoid to set ld_Library_path to locate the test libraries
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,@loader_path/.")
endif()

include_directories(${ROOT_INCLUDE_DIRS})
if(DEFINED ROOT_SOURCE_DIR)  # Testing using the binary tree
  include_directories(${ROOT_SOURCE_DIR}/tutorials)
  include_directories(${ROOT_SOURCE_DIR}/tmva/inc)
else()
  include_directories(${ROOT_INCLUDE_DIRS}/../tutorials)
endif()

#---environment-------------------------------------------------------------------------------
ROOT_ADD_TEST(show-environment COMMAND ${CMAKE_COMMAND} -E environment)

#---hworld------------------------------------------------------------------------------------
ROOT_EXECUTABLE(hworld hworld.cxx LIBRARIES Gpad)

#---event-------------------------------------------------------------------------------------
ROOT_GENERATE_DICTIONARY(EventDict ${CMAKE_CURRENT_SOURCE_DIR}/Event.h LINKDEF EventLinkDef.h)
ROOT_LINKER_LIBRARY(Event Event.cxx EventDict.cxx LIBRARIES Hist MathCore)

ROOT_EXECUTABLE(testevent MainEvent.cxx LIBRARIES Event RIO Tree Hist Net)
ROOT_ADD_TEST(test-event COMMAND testevent)

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
ROOT_GENERATE_DICTIONARY(HelloDict ${CMAKE_CURRENT_SOURCE_DIR}/Hello.h)
ROOT_LINKER_LIBRARY(Hello Hello.cxx HelloDict.cxx LIBRARIES Graf Gpad)

#--Aclockso------------------------------------------------------------------------------------
ROOT_GENERATE_DICTIONARY(AclockDict ${CMAKE_CURRENT_SOURCE_DIR}/Aclock.h)
ROOT_LINKER_LIBRARY(Aclock Aclock.cxx AclockDict.cxx LIBRARIES Graf Gpad)

#--bench------------------------------------------------------------------------------------
ROOT_GENERATE_DICTIONARY(TBenchDict ${CMAKE_CURRENT_SOURCE_DIR}/TBench.h LINKDEF benchLinkDef.h)
ROOT_LINKER_LIBRARY(TBench TBench.cxx TBenchDict.cxx LIBRARIES Core MathCore RIO Tree)
ROOT_EXECUTABLE(bench bench.cxx LIBRARIES Core TBench)
ROOT_ADD_TEST(test-bench COMMAND bench)

#--stress------------------------------------------------------------------------------------
ROOT_EXECUTABLE(stress stress.cxx LIBRARIES Event Core Hist RIO Tree Gpad Postscript)
ROOT_ADD_TEST(test-stress COMMAND stress -b FAILREGEX "FAILED|Error in"
                          DEPENDS test-event)

#--stressShapes------------------------------------------------------------------------------------
ROOT_EXECUTABLE(stressShapes stressShapes.cxx LIBRARIES  Geom Tree GenVector Gpad)
ROOT_ADD_TEST(test-stressshapes COMMAND stressShapes -b FAILREGEX "FAILED")

#--stressGeometry------------------------------------------------------------------------------------
ROOT_EXECUTABLE(stressGeometry stressGeometry.cxx LIBRARIES Geom Tree GenVector Gpad)
ROOT_ADD_TEST(test-stressgeometry COMMAND stressGeometry -b FAILREGEX "FAILED")

#--stressLinear------------------------------------------------------------------------------------
ROOT_EXECUTABLE(stressLinear stressLinear.cxx LIBRARIES Matrix Hist RIO)
ROOT_ADD_TEST(test-stresslinear COMMAND stressLinear FAILREGEX "FAILED")

#--stressGraphics------------------------------------------------------------------------------------
ROOT_EXECUTABLE(stressGraphics stressGraphics.cxx LIBRARIES Graf Gpad Postscript)
configure_file(stressGraphics.ref stressGraphics.ref COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../tutorials/graphics/earth.dat earth.dat @COPY_ONLY)
ROOT_ADD_TEST(test-stressgraphics COMMAND stressGraphics -b FAILREGEX "FAILED")

#--stressHistogram------------------------------------------------------------------------------------
ROOT_EXECUTABLE(stressHistogram stressHistogram.cxx LIBRARIES Hist RIO)
ROOT_ADD_TEST(test-stresshistogram COMMAND stressHistogram FAILREGEX "FAILED")

#--stressGUI---------------------------------------------------------------------------------------
ROOT_EXECUTABLE(stressGUI stressGUI.cxx LIBRARIES Gui Recorder GuiHtml ASImageGui)
#---Cannot run GUI test in batch mode--------------------
#configure_file(stressGUI.ref stressGUI.ref COPYONLY)
#ROOT_ADD_TEST(test-stressgui COMMAND stressGUI -ref FAILREGEX "FAILED")

#--stressSpectrum----------------------------------------------------------------------------------
ROOT_EXECUTABLE(stressSpectrum stressSpectrum.cxx LIBRARIES Hist Spectrum Gpad)
ROOT_ADD_TEST(test-stressspectrum COMMAND stressSpectrum -b FAILREGEX "FAILED")

#--stressVector------------------------------------------------------------------------------------
ROOT_EXECUTABLE(stressVector stressVector.cxx LIBRARIES Physics GenVector)
ROOT_ADD_TEST(test-stressvector COMMAND stressVector FAILREGEX "FAILED")

#--stressTMVA--------------------------------------------------------------------------------------
if(ROOT_tmva_FOUND)
  ROOT_EXECUTABLE(stressTMVA stressTMVA.cxx LIBRARIES TMVA)
  ROOT_ADD_TEST(test-stresstmva COMMAND stressTMVA -b)  
endif()

#--stressMathMore----------------------------------------------------------------------------------
if(ROOT_mathmore_FOUND)
  ROOT_EXECUTABLE(stressMathMore stressMathMore.cxx LIBRARIES MathMore)
  ROOT_ADD_TEST(test-stressmathmore COMMAND stressMathMore FAILREGEX "FAILED")  
endif()

#--stressMathCore----------------------------------------------------------------------------------
if(ROOT_mathcore_FOUND)
  ROOT_EXECUTABLE(stressMathCore stressMathCore.cxx LIBRARIES MathCore)
  ROOT_ADD_TEST(test-stressmathcore COMMAND stressMathCore FAILREGEX "FAILED")  
endif()

#--stressRooFit----------------------------------------------------------------------------------
if(ROOT_roofit_FOUND)
  ROOT_EXECUTABLE(stressRooFit stressRooFit.cxx LIBRARIES RooFit)
  ROOT_ADD_TEST(test-stressroofit COMMAND stressRooFit FAILREGEX "FAILED")  
endif()

#--stressRooStats----------------------------------------------------------------------------------
if(ROOT_roofit_FOUND)
  ROOT_EXECUTABLE(stressRooStats stressRooStats.cxx LIBRARIES RooStats)
  ROOT_ADD_TEST(test-stressroostats COMMAND stressRooStats -f ${CMAKE_SOURCE_DIR}/test/stressRooStats_ref.root FAILREGEX "FAILED")  
endif()

#--stressHistFactory--------------------------------------------------------------------------------
if(ROOT_roofit_FOUND AND ROOT_xml_FOUND AND NOT WIN32)
  ROOT_EXECUTABLE(stressHistFactory stressHistFactory.cxx LIBRARIES RooStats HistFactory XMLParser)
  file(COPY ${CMAKE_SOURCE_DIR}/test/HistFactoryTest.tar DESTINATION  ${CMAKE_CURRENT_BINARY_DIR} )
  ROOT_ADD_TEST(test-stressHistFactory ENVIRONMENT ROOTSYS=${CMAKE_BINARY_DIR} COMMAND stressHistFactory -f ${CMAKE_SOURCE_DIR}/test/stressHistFactory_ref.root FAILREGEX "FAILED")  
endif()

#--stressFit---------------------------------------------------------------------------------
ROOT_EXECUTABLE(stressFit stressFit.cxx LIBRARIES MathCore Matrix)
ROOT_ADD_TEST(test-stressfit COMMAND stressFit FAILREGEX "FAILED")

#--stressHistoFit-----------------------------------------------------------------------------
if(ROOT_unuran_FOUND)
  ROOT_EXECUTABLE(stressHistoFit stressHistoFit.cxx LIBRARIES MathCore Matrix Unuran Tree Gpad)
  ROOT_ADD_TEST(test-stresshistofit COMMAND stressHistoFit FAILREGEX "FAILED")
endif()

#--stressEntryList---------------------------------------------------------------------------
ROOT_EXECUTABLE(stressEntryList stressEntryList.cxx LIBRARIES MathCore Tree Hist)
ROOT_ADD_TEST(test-stressentrylist COMMAND stressEntryList -b FAILREGEX "FAILED")

#--stressIterators---------------------------------------------------------------------------
ROOT_EXECUTABLE(stressIterators stressIterators.cxx LIBRARIES Core)
ROOT_ADD_TEST(test-stressiterators COMMAND stressIterators FAILREGEX "FAILED")

#--stressInterpreter-------------------------------------------------------------------------
ROOT_EXECUTABLE(stressInterpreter stressInterpreter.cxx LIBRARIES Core)
if(WIN32)
  set(cleantmpfiles POSTCMD cmd /c del AutoDict_*)
  set(setenvironment "")
else()
  set(setenvironment ENVIRONMENT PATH=${CMAKE_BINARY_DIR}/bin${cfgdir}:$ENV{PATH})
endif()
ROOT_ADD_TEST(test-stressinterpreter COMMAND stressInterpreter
              PRECMD ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/stressInterpreter.cxx 
                                              $<TARGET_FILE_DIR:stressInterpreter>/stressInterpreter.cxx
              ${cleantmpfiles}
              ${setenvironment}
              FAILREGEX "FAILED|Error")

#--stressHepix-------------------------------------------------------------------------------
#ROOT_EXECUTABLE(stressHepix stressHepix.cxx LIBRARIES Core)
#ROOT_ADD_TEST(test-stressHepix COMMAND stressHepix FAILREGEX "FAILED")

#--stressProof-------------------------------------------------------------------------------
if(NOT WIN32 AND ROOT_xrootd_FOUND)
  add_custom_target(TestData ALL COMMAND ${CMAKE_COMMAND} -DDST=${CMAKE_SOURCE_DIR}/files -P ${CMAKE_CURRENT_SOURCE_DIR}/rootDownloadData.cmake)  
  ROOT_EXECUTABLE(stressProof stressProof.cxx LIBRARIES Proof ProofPlayer Hist)
  ROOT_ADD_TEST(test-stressproof COMMAND stressProof lite:// 
                                         -h1 ${CMAKE_SOURCE_DIR}/files/h1 -event ${CMAKE_SOURCE_DIR}/files/event
                                         -l /tmp/stressProof-%d.log -cleanlog -catlog -noprogress
                                         ENVIRONMENT PATH=${CMAKE_BINARY_DIR}/bin:$ENV{PATH}
                                         FAILREGEX "FAILED")  
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
