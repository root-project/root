############################################################################
# CMakeLists.txt file for building ROOT gui/guihtml package
############################################################################

ROOT_USE_PACKAGE(gui/gui)
ROOT_USE_PACKAGE(net/net)

ROOT_STANDARD_LIBRARY_PACKAGE(GuiHtml DEPENDENCIES Gui Graf Net)

