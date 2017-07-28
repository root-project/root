############################################################################
# CMakeLists.txt file for building ROOT rootfit package
# @author Pere Mato, CERN
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(RooFit
                              DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                              LINKDEF LinkDef1.h
                              LIBRARIES ${ROOT_MATHMORE_LIBRARY}
                              DEPENDENCIES Core RooFitCore Tree RIO Matrix MathCore)
