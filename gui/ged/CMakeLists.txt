############################################################################
# CMakeLists.txt file for building ROOT gui/ged package
############################################################################

ROOT_USE_PACKAGE(gui/gui)
ROOT_USE_PACKAGE(tree/treeplayer)

ROOT_STANDARD_LIBRARY_PACKAGE(Ged DEPENDENCIES Hist Gpad Graf Gui Tree TreePlayer)

