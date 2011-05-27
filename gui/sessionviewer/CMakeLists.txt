############################################################################
# CMakeLists.txt file for building ROOT gui/sessionviewer package
############################################################################

ROOT_USE_PACKAGE(gui/gui)
ROOT_USE_PACKAGE(tree/treeplayer)
ROOT_USE_PACKAGE(proof/proof)

ROOT_STANDARD_LIBRARY_PACKAGE(SessionViewer DEPENDENCIES Proof Gui Hist Gpad Graf Tree MathCore)


