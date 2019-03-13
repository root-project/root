############################################################################
# CMakeLists.txt file for building ROOT gui/guihtml package
############################################################################

if(ssl)
  add_definitions(-DR__SSL)
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(GuiHtml
  HEADERS
    TGHtml.h
    TGHtmlBrowser.h
    TGHtmlTokens.h
    TGHtmlUri.h
  SOURCES
    src/TGHtml.cxx
    src/TGHtmlBrowser.cxx
    src/TGHtmlDraw.cxx
    src/TGHtmlElement.cxx
    src/TGHtmlForm.cxx
    src/TGHtmlImage.cxx
    src/TGHtmlIndex.cxx
    src/TGHtmlLayout.cxx
    src/TGHtmlParse.cxx
    src/TGHtmlSizer.cxx
    src/TGHtmlTable.cxx
    src/TGHtmlTokenMap.cxx
    src/TGHtmlUri.cxx
  DEPENDENCIES
    Gui
    Graf
    Net
)
