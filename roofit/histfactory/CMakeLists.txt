############################################################################
# CMakeLists.txt file for building ROOT rootfit/histfactory package
# @author Pere Mato, CERN
############################################################################

ROOT_GENERATE_DICTIONARY(G__HistFactory RooStats/HistFactory/*.h MODULE HistFactory LINKDEF LinkDef.h OPTIONS "-writeEmptyRootPCM" BUILTINS GSL)


ROOT_GLOB_SOURCES(sources src/*.cxx)
list(REMOVE_ITEM sources ${CMAKE_SOURCE_DIR}/roofit/histfactory/src/hist2workspace.cxx)

ROOT_LINKER_LIBRARY(HistFactory ${sources} G__HistFactory.cxx LIBRARIES Core 
                                DEPENDENCIES RooFit RooFitCore Tree RIO Hist Matrix MathCore Minuit Foam Graf Gpad RooStats XMLParser)
ROOT_EXECUTABLE(hist2workspace MakeModelAndMeasurements.cxx hist2workspace.cxx
			       LIBRARIES HistFactory RooFit RooFitCore Tree RIO Matrix Hist ${ROOT_MATHMORE_LIBRARY} ${GSL_LIBRARIES} MathCore 
                                         Graf Gpad Minuit Foam RooStats XMLParser
                               BUILTINS GSL)
                                         

file(COPY config/prepareHistFactory DESTINATION  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
install(FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/prepareHistFactory
                PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                            GROUP_EXECUTE GROUP_READ
                            WORLD_EXECUTE WORLD_READ
                DESTINATION ${CMAKE_INSTALL_BINDIR})

ROOT_INSTALL_HEADERS()
