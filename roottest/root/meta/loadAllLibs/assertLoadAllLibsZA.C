#include "libraryLister.h"

int assertLoadAllLibsZA(){

  gSystem->Setenv("DISPLAY",""); // avoid spurrious warning when loading libGui

  //outputRAII o;

  gInterpreter->SetClassAutoparsing(false);

  auto libList = getLibrariesList();
  libList.sort();
  libList.reverse();
  libList.unique();
  //libList.remove_if([](std::string){ static int n = 0; return (n++ % 2 == 0);});
  loadLibrariesInList(libList);

  return 0;
}

