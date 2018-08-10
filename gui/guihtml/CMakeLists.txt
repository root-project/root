############################################################################
# CMakeLists.txt file for building ROOT gui/guihtml package
############################################################################

if(ssl)
  add_definitions(-DR__SSL)
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(GuiHtml DEPENDENCIES Gui Graf Net)
