# Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.
#
# Author: Pere Mato, 25/10/2010

# Testing using the binary tree
set(root_exe $<TARGET_FILE:root.exe>)

#---environment-------------------------------------------------------------------------------
ROOT_ADD_TEST(show-environment COMMAND ${CMAKE_COMMAND} -E environment)

#---hworld------------------------------------------------------------------------------------
ROOT_EXECUTABLE(hworld hworld.cxx LIBRARIES Gpad)

#---event-------------------------------------------------------------------------------------
ROOT_STANDARD_LIBRARY_PACKAGE(Event
                              NO_INSTALL_HEADERS
                              HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/Event.h
                              SOURCES Event.cxx LINKDEF EventLinkDef.h
                              DEPENDENCIES Hist MathCore)
if(MSVC AND NOT CMAKE_GENERATOR MATCHES Ninja)
  add_custom_command(TARGET Event POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/libEvent.dll
                                     ${CMAKE_CURRENT_BINARY_DIR}/libEvent.dll)
  if(NOT runtime_cxxmodules)
    add_custom_command(TARGET Event POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/libEvent_rdict.pcm
                                       ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/libEvent_rdict.pcm)
  else()
    add_custom_command(TARGET Event POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/Event.pcm
                                     ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/Event.pcm)
  endif()
endif()
ROOT_EXECUTABLE(eventexe MainEvent.cxx LIBRARIES Event RIO Tree TreePlayer Hist Net)
ROOT_ADD_TEST(test-event COMMAND eventexe)

#---guitest------------------------------------------------------------------------------------
if(TARGET Gui)
  ROOT_EXECUTABLE(guitest guitest.cxx LIBRARIES RIO Gui Gpad Hist)
endif()

#---hsimple------------------------------------------------------------------------------------
#ROOT_EXECUTABLE(hsimple hsimple.cxx LIBRARIES RIO Tree Hist)
#ROOT_ADD_TEST(test-hsimple COMMAND hsimple)

#---invalid pointer checks---------------------------------------------------------------------
if((NOT (APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES arm64) OR M1_BROKEN_TESTS) AND NOT experimental_adaptivecpp)
  ROOT_ADD_TEST(test-check-nullptr
    COMMAND ${root_exe} -b -q -l --ptrcheck -e "int*ptr=(int*)0" -e "*ptr"
    PASSREGEX "dereference null pointer")

  if(NOT MSVC OR win_broken_tests)
    ROOT_ADD_TEST(test-check-invalidptr
      COMMAND ${root_exe} -b -q -l --ptrcheck -e "int*ptr=(int*)1" -e "*ptr"
      PASSREGEX "invalid memory address")
  endif()
endif()

#--minexam-------------------------------------------------------------------------------------
ROOT_EXECUTABLE(minexam minexam.cxx LIBRARIES Hist)
ROOT_ADD_TEST(test-minexam COMMAND minexam)

#--tstring-------------------------------------------------------------------------------------
ROOT_EXECUTABLE(tstring tstring.cxx LIBRARIES Core)
ROOT_ADD_TEST(test-tstring COMMAND tstring)

#--tcollex-------------------------------------------------------------------------------------
ROOT_EXECUTABLE(tcollex tcollex.cxx LIBRARIES Core)
ROOT_ADD_TEST(test-tcollex COMMAND tcollex)

#--tcollbm------------------------------------------------------------------------------------
ROOT_EXECUTABLE(tcollbm tcollbm.cxx LIBRARIES Core MathCore)
ROOT_ADD_TEST(test-tcollbm COMMAND tcollbm 1000 1000000 LABELS longtest)

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
ROOT_GENERATE_DICTIONARY(G__Hello ${CMAKE_CURRENT_SOURCE_DIR}/Hello.h MODULE Hello DEPENDENCIES Gpad Graf Hist MathCore Matrix)
ROOT_LINKER_LIBRARY(Hello Hello.cxx G__Hello.cxx LIBRARIES Graf Gpad)

#--Aclockso------------------------------------------------------------------------------------
ROOT_GENERATE_DICTIONARY(G__Aclock ${CMAKE_CURRENT_SOURCE_DIR}/Aclock.h MODULE Aclock DEPENDENCIES Graf Gpad MathCore)
ROOT_LINKER_LIBRARY(Aclock Aclock.cxx G__Aclock.cxx LIBRARIES Graf Gpad)

#--bench------------------------------------------------------------------------------------
ROOT_GENERATE_DICTIONARY(G__TBench ${CMAKE_CURRENT_SOURCE_DIR}/TBench.h MODULE TBench LINKDEF benchLinkDef.h DEPENDENCIES MathCore Tree)
ROOT_LINKER_LIBRARY(TBench TBench.cxx G__TBench.cxx LIBRARIES Core MathCore RIO Tree)
ROOT_EXECUTABLE(bench bench.cxx LIBRARIES Core TBench RIO)
ROOT_ADD_TEST(test-bench COMMAND bench -s LABELS longtest)

#--stress------------------------------------------------------------------------------------
  ROOT_EXECUTABLE(stress stress.cxx LIBRARIES Event Core Hist RIO Tree Gpad Postscript)
  ROOT_ADD_TEST(test-stress COMMAND stress -b FAILREGEX "FAILED|Error in"
                            DEPENDS test-event LABELS longtest)

if(geom)
  #--stressShapes------------------------------------------------------------------------------------
  ROOT_EXECUTABLE(stressShapes stressShapes.cxx LIBRARIES  Geom Tree GenVector Gpad)
  ROOT_ADD_TEST(test-stressshapes COMMAND stressShapes -b FAILREGEX "FAILED|Error in")
  ROOT_ADD_TEST(test-stressshapes-interpreted COMMAND ${root_exe} -b -q -l ${CMAKE_CURRENT_SOURCE_DIR}/stressShapes.cxx
                FAILREGEX "FAILED|Error in")

  #--stressGeometry------------------------------------------------------------------------------------
  if (NOT MSVC) # Prevents the access to the web, which could happen through https
    ROOT_EXECUTABLE(stressGeometry stressGeometry.cxx LIBRARIES Geom Tree GenVector Gpad)
    ROOT_ADD_TEST(test-stressgeometry COMMAND stressGeometry -b FAILREGEX "FAILED|Error in" LABELS longtest)
    ROOT_ADD_TEST(test-stressgeometry-interpreted COMMAND ${root_exe} -b -q -l ${CMAKE_CURRENT_SOURCE_DIR}/stressGeometry.cxx
                  FAILREGEX "FAILED|Error in" DEPENDS test-stressgeometry LABELS longtest)
  endif()
endif()

#--stressLinear------------------------------------------------------------------------------------
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/linearIO.root ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
ROOT_EXECUTABLE(stressLinear stressLinear.cxx LIBRARIES Matrix Hist RIO)
ROOT_ADD_TEST(test-stresslinear COMMAND stressLinear FAILREGEX "FAILED|Error in" LABELS longtest)
ROOT_ADD_TEST(test-stresslinear-interpreted COMMAND ${root_exe} -b -q -l ${CMAKE_CURRENT_SOURCE_DIR}/stressLinear.cxx
              FAILREGEX "FAILED|Error in" LABELS longtest)

#--stressGraphics------------------------------------------------------------------------------------
if(opengl AND TARGET TreeViewer)
  ROOT_EXECUTABLE(stressGraphics stressGraphics.cxx LIBRARIES Graf Gpad Postscript TreeViewer HistPainter)
  if(MSVC)
    set_property(TARGET stressGraphics APPEND_STRING PROPERTY LINK_FLAGS "/STACK:4000000")
  endif()
  # with ZLIB-CF(builtin_zlib) we have better compression ratio.
  if(ZLIB_CF)
    configure_file(stressGraphics_builtinzlib.ref stressGraphics_builtinzlib.ref COPYONLY)
  else()
    configure_file(stressGraphics.ref stressGraphics.ref COPYONLY)
  endif()
  configure_file(stressGraphics_web.ref stressGraphics_web.ref COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../tutorials/visualisation/graphics/earth.dat earth.dat COPYONLY)
  ROOT_ADD_TEST(test-stressgraphics
                ENVIRONMENT LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}
                COMMAND stressGraphics -b --web=off -k -p=sge
                FAILREGEX "FAILED|Error in"
                LABELS longtest)
  ROOT_ADD_TEST(test-stressgraphics-interpreted
                COMMAND ${root_exe} -b --web=off -q -l ${CMAKE_CURRENT_SOURCE_DIR}/stressGraphics.cxx
                FAILREGEX "FAILED|Error in"
                DEPENDS test-stressgraphics)
  ROOT_ADD_TEST(test-stressgraphics-svg
                ENVIRONMENT LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}
                COMMAND stressGraphics -b --web=off -p=svg --svg=${CMAKE_CURRENT_SOURCE_DIR}/svg_ref/
                FAILREGEX "FAILED|Error in"
                LABELS longtest)
  if(CHROME_EXECUTABLE)
    ROOT_ADD_TEST(test-stressgraphics-chrome
                  RUN_SERIAL
                  ENVIRONMENT LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}
                  COMMAND stressGraphics -b -k -p=sgc --web=chrome
                  FAILREGEX "FAILED|Error in"
                  LABELS longtest)
  endif()
  if(FIREFOX_EXECUTABLE)
    ROOT_ADD_TEST(test-stressgraphics-firefox-skip3d
                  RUN_SERIAL
                  ENVIRONMENT LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}
                  COMMAND stressGraphics -b -k -p=sgf --web=firefox -skip3d
                  FAILREGEX "FAILED|Error in"
                  LABELS longtest)
  endif()
endif()

#--stressHistogram------------------------------------------------------------------------------------
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/stressHistogram.5.18.00.root ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
ROOT_EXECUTABLE(stressHistogram stressHistogram.cxx LIBRARIES Hist RIO)
ROOT_ADD_TEST(test-stresshistogram COMMAND stressHistogram FAILREGEX "FAILED|Error in" LABELS longtest)
ROOT_ADD_TEST(test-stresshistogram-interpreted COMMAND ${root_exe} -b -q -l ${CMAKE_CURRENT_SOURCE_DIR}/stressHistogram.cxx
              FAILREGEX "FAILED|Error in")

#--stressGUI---------------------------------------------------------------------------------------
if(asimage)
  ROOT_EXECUTABLE(stressGUI stressGUI.cxx LIBRARIES Gui Recorder GuiHtml ASImageGui)
endif()

#--stressSpectrum----------------------------------------------------------------------------------
if (spectrum)
  ROOT_EXECUTABLE(stressSpectrum stressSpectrum.cxx LIBRARIES Hist Spectrum Gpad)
  ROOT_ADD_TEST(test-stressspectrum COMMAND stressSpectrum -b FAILREGEX "FAILED|Error in" LABELS longtest)
  ROOT_ADD_TEST(test-stressspectrum-interpreted COMMAND ${root_exe} -b -q -l ${CMAKE_CURRENT_SOURCE_DIR}/stressSpectrum.cxx
                FAILREGEX "FAILED|Error in" DEPENDS test-stressspectrum LABELS longtest)
endif()

#--stressVector------------------------------------------------------------------------------------
ROOT_EXECUTABLE(stressVector stressVector.cxx LIBRARIES Physics GenVector Smatrix)
ROOT_ADD_TEST(test-stressvector COMMAND stressVector FAILREGEX "FAILED|Error in")
ROOT_ADD_TEST(test-stressvector-interpreted COMMAND ${root_exe} -b -q -l ${CMAKE_CURRENT_SOURCE_DIR}/stressVector.cxx
              FAILREGEX "FAILED|Error in" DEPENDS test-stressvector)

#--stressMathMore----------------------------------------------------------------------------------
if(mathmore)
  ROOT_EXECUTABLE(stressMathMore stressMathMore.cxx LIBRARIES MathMore Smatrix)
  ROOT_ADD_TEST(test-stressmathmore COMMAND stressMathMore FAILREGEX "FAILED|Error in" LABELS longtest)
  ROOT_ADD_TEST(test-stressmathmore-interpreted COMMAND ${root_exe} -b -q -l ${CMAKE_CURRENT_SOURCE_DIR}/stressMathMore.cxx
                FAILREGEX "FAILED|Error in" DEPENDS test-stressmathmore TIMEOUT 1800)
endif()

#--stressMathCore----------------------------------------------------------------------------------
ROOT_STANDARD_LIBRARY_PACKAGE(TrackMathCoreDict
                              NO_SOURCES
                              NO_INSTALL_HEADERS
                              HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/TrackMathCore.h
                              LINKDEF TrackMathCoreLinkDef.h
                              DEPENDENCIES Core MathCore RIO GenVector Smatrix)
if(MSVC AND NOT CMAKE_GENERATOR MATCHES Ninja)
  add_custom_command(TARGET TrackMathCoreDict POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/libTrackMathCoreDict.dll
                                     ${CMAKE_CURRENT_BINARY_DIR}/libTrackMathCoreDict.dll)
endif()
ROOT_EXECUTABLE(stressMathCore stressMathCore.cxx LIBRARIES MathCore Hist RIO Tree GenVector Smatrix)
ROOT_ADD_TEST(test-stressmathcore COMMAND stressMathCore FAILREGEX "FAILED|Error in" LABELS longtest)
ROOT_ADD_TEST(test-stressmathcore-interpreted COMMAND ${root_exe} -b -q -l ${CMAKE_CURRENT_SOURCE_DIR}/stressMathCore.cxx
              FAILREGEX "FAILED|Error in" DEPENDS test-stressmathcore)

#--stressFit---------------------------------------------------------------------------------
ROOT_EXECUTABLE(stressFit stressFit.cxx LIBRARIES MathCore Matrix)
ROOT_ADD_TEST(test-stressfit COMMAND stressFit FAILREGEX "FAILED|Error in" LABELS longtest)
ROOT_ADD_TEST(test-stressfit-interpreted COMMAND ${root_exe} -b -q -l ${CMAKE_CURRENT_SOURCE_DIR}/stressFit.cxx
              FAILREGEX "FAILED|Error in" DEPENDS test-stressfit)

#--stressHistoFit-----------------------------------------------------------------------------
if(unuran)
  ROOT_EXECUTABLE(stressHistoFit stressHistoFit.cxx LIBRARIES MathCore Matrix Unuran Tree Gpad)
  ROOT_ADD_TEST(test-stresshistofit COMMAND stressHistoFit FAILREGEX "FAILED|Error in" LABELS longtest)
  if(NOT MSVC OR llvm13_broken_tests)
    ROOT_ADD_TEST(test-stresshistofit-interpreted COMMAND ${root_exe} -b -q -l ${CMAKE_CURRENT_SOURCE_DIR}/stressHistoFit.cxx
                  FAILREGEX "FAILED|Error in" DEPENDS test-stresshistofit )
  endif()
endif()

#--stressEntryList---------------------------------------------------------------------------
ROOT_EXECUTABLE(stressEntryList stressEntryList.cxx LIBRARIES MathCore Tree Hist)
ROOT_ADD_TEST(test-stressentrylist COMMAND stressEntryList -b FAILREGEX "FAILED|Error in")
ROOT_ADD_TEST(test-stressentrylist-interpreted COMMAND ${root_exe} -b -q -l ${CMAKE_CURRENT_SOURCE_DIR}/stressEntryList.cxx
              FAILREGEX "FAILED|Error in" DEPENDS test-stressentrylist)

#--stressIterators---------------------------------------------------------------------------
ROOT_EXECUTABLE(stressIterators stressIterators.cxx LIBRARIES Core)
ROOT_ADD_TEST(test-stressiterators COMMAND stressIterators FAILREGEX "FAILED|Error in")
ROOT_ADD_TEST(test-stressiterators-interpreted COMMAND ${root_exe} -b -q -l ${CMAKE_CURRENT_SOURCE_DIR}/stressIterators.cxx
              FAILREGEX "FAILED|Error in" DEPENDS test-stressiterators)

#--stressInterpreter-------------------------------------------------------------------------
ROOT_EXECUTABLE(stressInterpreter stressInterpreter.cxx LIBRARIES Core)
if(WIN32)
  set(cleantmpfiles POSTCMD cmd /c del AutoDict_*)
endif()
ROOT_ADD_TEST(test-stressinterpreter COMMAND stressInterpreter
              WORKING_DIR ${CMAKE_BINARY_DIR}
              PRECMD ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/stressInterpreter.cxx
                                              $<TARGET_FILE_DIR:stressInterpreter>/stressInterpreter.cxx
              ${cleantmpfiles}
              FAILREGEX Error|FAILED)

#--stressHepix-------------------------------------------------------------------------------
ROOT_EXECUTABLE(stressHepix stressHepix.cxx LIBRARIES Core)
#ROOT_ADD_TEST(test-stressHepix COMMAND stressHepix FAILREGEX "FAILED|Error in")

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
if(xrootd)
  ROOT_ADD_TEST(test-stressIOPlugins-xroot COMMAND stressIOPlugins root FAILREGEX "FAILED|Error in")
endif()
if(davix)
  ROOT_ADD_TEST(test-stressIOPlugins-http COMMAND stressIOPlugins http FAILREGEX "FAILED|Error in")
endif()

#--delaunay----------------------------------------------------------------------------------
ROOT_EXECUTABLE(delaunayTriangulation delaunayTriangulation.cxx LIBRARIES Hist)
ROOT_ADD_TEST(test-delaunay COMMAND delaunayTriangulation)

#--TestTformula------------------------------------------------------------------------------------
ROOT_EXECUTABLE(TFormulaTests TFormulaTests.cxx LIBRARIES Hist RIO)
if(MSVC)
  set_property(TARGET TFormulaTests APPEND_STRING PROPERTY LINK_FLAGS "/STACK:4000000")
endif()
ROOT_ADD_TEST(test-TFormulaTests COMMAND TFormulaTests FAILREGEX "FAILED|Error in")

#--std::simd GenVector test-----------------------------------------------------------------------------------
if(ROOT_HAVE_EXPERIMENTAL_SIMD)
  ROOT_EXECUTABLE(testGenVectorSimd testGenVectorSimd.cxx LIBRARIES Physics GenVector)
  ROOT_ADD_TEST(test-GenVector-Simd COMMAND testGenVectorSimd FAILREGEX "FAILED|Error in")
endif()

#--g2root------------------------------------------------------------------------------------------
if(TARGET g2root)
  ROOT_ADD_TEST(test-g2root COMMAND g2root)
endif()

#--canary tests------------------------------------------------------------------------------------
if(asserts AND NOT MSVC)
  ROOT_EXECUTABLE(checkAssertsNDEBUG checkAssertsNDEBUG.cxx LIBRARIES Core)
  ROOT_ADD_TEST(checkAssertsNDEBUG COMMAND checkAssertsNDEBUG)
endif()
