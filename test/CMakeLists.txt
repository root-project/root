# CMakeLists.txt for the ROOT test programs.
# This shows nicely how to compile and link applications
# using the ROOT libraries on all supported platforms.
#
# Author: Pere Mato, 25/10/2010
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(test)
find_package(ROOT REQUIRED)

include(${ROOT_USE_FILE})

include_directories(${ROOT_INCLUDE_DIRS})

#---hworld------------------------------------------------------------------------------------
ROOT_EXECUTABLE(hworld hworld.cxx LIBRARIES Gpad)

#---event-------------------------------------------------------------------------------------
ROOT_GENERATE_DICTIONARY(EventDict ${CMAKE_CURRENT_SOURCE_DIR}/Event.h LINKDEF EventLinkDef.h)
ROOT_LINKER_LIBRARY(Event Event.cxx EventDict.cxx LIBRARIES Hist MathCore)

ROOT_EXECUTABLE(eventexe MainEvent.cxx LIBRARIES Event RIO Tree Hist Net)
ROOT_ADD_TEST(test-event COMMAND eventexe)

#---hsimple------------------------------------------------------------------------------------
ROOT_EXECUTABLE(hsimple hsimple.cxx LIBRARIES RIO Tree Hist)
ROOT_ADD_TEST(test-hsimple COMMAND hsimple)

#--minexam-------------------------------------------------------------------------------------
ROOT_EXECUTABLE(minexam minexam.cxx LIBRARIES Minuit)
ROOT_ADD_TEST(test-minexam COMMAND minexam)

#--tstring-------------------------------------------------------------------------------------
ROOT_EXECUTABLE(tstring tstring.cxx LIBRARIES Core)
configure_file(tstring.cxx tstring.cxx @COPY_ONLY)
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
ROOT_ADD_TEST(test-stress COMMAND stress -b FAILREGEX "FAILED")

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
configure_file(stressGraphics.ref stressGraphics.ref @COPY_ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../tutorials/graphics/earth.dat earth.dat @COPY_ONLY)
ROOT_ADD_TEST(test-stressgraphics COMMAND stressGraphics -b FAILREGEX "FAILED")

#--stressHistogram------------------------------------------------------------------------------------
ROOT_EXECUTABLE(stressHistogram stressHistogram.cxx LIBRARIES Hist RIO)
ROOT_ADD_TEST(test-stresshistogram COMMAND stressHistogram FAILREGEX "FAILED")

#--stressGUI---------------------------------------------------------------------------------------
ROOT_EXECUTABLE(stressGUI stressGUI.cxx LIBRARIES Gui Recorder GuiHtml ASImageGui)
#configure_file(stressGUI.ref stressGUI.ref @COPY_ONLY)
ROOT_ADD_TEST(test-stressgui COMMAND stressGUI -b -ref FAILREGEX "FAILED")

#--stressSpectrum----------------------------------------------------------------------------------
ROOT_EXECUTABLE(stressSpectrum stressSpectrum.cxx LIBRARIES Hist Spectrum Gpad)
ROOT_ADD_TEST(test-stressspectrum COMMAND stressSpectrum -b FAILREGEX "FAILED")

#--stressVector------------------------------------------------------------------------------------
ROOT_EXECUTABLE(stressVector stressVector.cxx LIBRARIES Physics GenVector)
ROOT_ADD_TEST(test-stressvector COMMAND stressVector FAILREGEX "FAILED")

#--stressTMVA--------------------------------------------------------------------------------------
if(ROOT_tmva_FOUND)
  #  ROOT_EXECUTABLE(stressTMVA stressTMVA.cxx LIBRARIES TMVA)
  #  ROOT_ADD_TEST(test-stresstmva COMMAND stressTMVA)  
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
  #TODO-- Need to resolve the fact that RootFit not 'exported' due to allow the possibility to build standalone 
  ROOT_EXECUTABLE(stressRooFit stressRooFit.cxx LIBRARIES RooFit)
  ROOT_ADD_TEST(test-stressroofit COMMAND stressRooFit FAILREGEX "FAILED")  
endif()

#--stressFit----------------------------------------------------------------------------------------
ROOT_EXECUTABLE(stressFit stressFit.cxx LIBRARIES MathCore Matrix)
ROOT_ADD_TEST(test-stressfit COMMAND stressFit FAILREGEX "FAILED")

#--stressHitoFit----------------------------------------------------------------------------------------
if(ROOT_unuran_FOUND)
  ROOT_EXECUTABLE(stressHistoFit stressHistoFit.cxx LIBRARIES MathCore Matrix Unuran Tree Gpad)
  ROOT_ADD_TEST(test-stresshistofit COMMAND stressHistoFit FAILREGEX "FAILED")
endif()

#--stressEntryList----------------------------------------------------------------------------------------
ROOT_EXECUTABLE(stressEntryList stressEntryList.cxx LIBRARIES MathCore Tree Hist)
ROOT_ADD_TEST(test-stressentrylist COMMAND stressEntryList -b FAILREGEX "FAILED")

#--stressIterators----------------------------------------------------------------------------------------
ROOT_EXECUTABLE(stressIterators stressIterators.cxx LIBRARIES Core)
ROOT_ADD_TEST(test-stressiterators COMMAND stressIterators FAILREGEX "FAILED")

#--stressInterpreter----------------------------------------------------------------------------------------
ROOT_EXECUTABLE(stressInterpreter stressInterpreter.cxx LIBRARIES Core)
configure_file(stressInterpreter.cxx ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/stressInterpreter.cxx @COPY_ONLY)
ROOT_ADD_TEST(test-stressinterpreter COMMAND stressInterpreter 
              FAILREGEX "FAILED|Error"
              ENVIRONMENT ${ld_library_path}=${ROOTSYS}/lib)

#--stressHepix----------------------------------------------------------------------------------------
#ROOT_EXECUTABLE(stressHepix stressHepix.cxx LIBRARIES Core)
#ROOT_ADD_TEST(test-stressHepix COMMAND stressHepix FAILREGEX "FAILED")

#--stressProof----------------------------------------------------------------------------------
if(ROOT_proof_FOUND)
  ROOT_EXECUTABLE(stressProof stressProof.cxx LIBRARIES Proof)
  ROOT_ADD_TEST(test-stressproof COMMAND stressProof FAILREGEX "FAILED")  
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





