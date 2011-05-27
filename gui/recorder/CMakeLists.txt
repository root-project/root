############################################################################
# CMakeLists.txt file for building ROOT gui/recorder package
############################################################################

ROOT_USE_PACKAGE(gui/gui)
ROOT_USE_PACKAGE(tree/tree)

ROOT_STANDARD_LIBRARY_PACKAGE(Recorder DEPENDENCIES RIO Tree Gui Thread)

