#include "libraryLister.h"

void execLoadAllLibsAZ(){

  outputRAII o;

  gInterpreter->SetClassAutoparsing(false);

  auto libList = getLibrariesList();
  libList.sort();
  loadLibrariesInList(libList);

}
