#include "libraryLister.h"

void execLoadAllLibsZA(){

  outputRAII o;

  gInterpreter->SetClassAutoparsing(false);

  auto libList = getLibrariesList();
  libList.sort();
  libList.reverse();
  loadLibrariesInList(libList);

}

