#include "libraryLister.h"

int assertLoadAllLibsAZ(){

  gSystem->Setenv("DISPLAY",""); // avoid spurrious warning when loading libGui

  //outputRAII o;

  gInterpreter->SetClassAutoparsing(false);

  auto libList = getLibrariesList();
  libList.sort();
  libList.unique();
  loadLibrariesInList(libList);

  return 0;
}
