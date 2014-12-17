#include "libraryLister.h"

void execLoadAllLibsZA(){

  gSystem->Setenv("DISPLAY",""); // avoid spurrious warning when loading libGui

  outputRAII o;

  gInterpreter->SetClassAutoparsing(false);

  auto libList = getLibrariesList();
  libList.sort();
  libList.reverse();
  loadLibrariesInList(libList);

}

