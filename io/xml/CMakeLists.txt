############################################################################
# CMakeLists.txt file for building ROOT io/xml package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(XMLIO
                              HEADERS
                                 TBufferXML.h
                                 TKeyXML.h
                                 TXMLEngine.h
                                 TXMLFile.h
                                 TXMLPlayer.h
                                 TXMLSetup.h
                              SOURCES
                                 src/TBufferXML.cxx
                                 src/TKeyXML.cxx
                                 src/TXMLEngine.cxx
                                 src/TXMLFile.cxx
                                 src/TXMLPlayer.cxx
                                 src/TXMLSetup.cxx
                              DEPENDENCIES RIO)
