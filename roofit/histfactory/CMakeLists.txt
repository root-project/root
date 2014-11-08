############################################################################
# CMakeLists.txt file for building ROOT rootfit/histfactory package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(io/xmlparser)

ROOT_GENERATE_DICTIONARY(G__HistFactory RooStats/HistFactory/*.h MODULE HistFactory LINKDEF LinkDef.h OPTIONS "-writeEmptyRootPCM")


                                               
ROOT_LINKER_LIBRARY(HistFactory  *.cxx G__HistFactory.cxx LIBRARIES Core 
                                DEPENDENCIES RooFit RooFitCore Tree RIO Hist Matrix MathCore Minuit Foam Graf Gpad RooStats XMLParser)
ROOT_EXECUTABLE(hist2workspace MakeModelAndMeasurements.cxx hist2workspace.cxx
			       LIBRARIES HistFactory RooFit RooFitCore Tree RIO Matrix Hist ${ROOT_MATHMORE_LIBRARY} ${GSL_LIBRARIES} MathCore 
                                         Graf Gpad Minuit Foam RooStats XMLParser)
                                         

if(builtin_gsl)
  ROOT_ADD_BUILTIN_DEPENDENCIES(hist2workspace GSL)
endif()

file(COPY config/prepareHistFactory DESTINATION  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
install(FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/prepareHistFactory
                PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                            GROUP_EXECUTE GROUP_READ
                            WORLD_EXECUTE WORLD_READ
                DESTINATION ${CMAKE_INSTALL_BINDIR})


#ROOT_INSTALL_HEADERS()
install(DIRECTORY inc/RooStats/HistFactory/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/RooStats/HistFactory 
                       COMPONENT headers
                       PATTERN ".svn" EXCLUDE
                       REGEX "LinkDef" EXCLUDE )
set_property(GLOBAL APPEND PROPERTY ROOT_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/inc)
