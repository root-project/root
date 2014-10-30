############################################################################
# CMakeLists.txt file for building ROOT roostats package
# @author Pere Mato, CERN
############################################################################

ROOT_GENERATE_DICTIONARY(G__RooStats RooStats/*.h MODULE RooStats LINKDEF LinkDef.h OPTIONS "-writeEmptyRootPCM")

ROOT_LINKER_LIBRARY(RooStats  *.cxx G__RooStats.cxx LIBRARIES Core 
                               DEPENDENCIES RooFit RooFitCore Tree RIO Hist Matrix MathCore Minuit Foam Graf Gpad )

#ROOT_INSTALL_HEADERS()
install(DIRECTORY inc/RooStats/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/RooStats
                       COMPONENT headers
                       PATTERN ".svn" EXCLUDE
                       REGEX "LinkDef" EXCLUDE )
set_property(GLOBAL APPEND PROPERTY ROOT_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/inc)
