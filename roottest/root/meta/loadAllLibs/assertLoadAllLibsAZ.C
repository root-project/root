#include "libraryLister.h"

int assertLoadAllLibsAZ(){

  gSystem->Setenv("DISPLAY",""); // avoid spurrious warning when loading libGui

  //outputRAII o;

  gInterpreter->SetClassAutoparsing(false);

  auto libList = getLibrariesList();
  libList.sort();
  libList.unique();
  //libList.remove_if([](std::string){ static int n = 0; return (n++ % 2 == 1);});
  loadLibrariesInList(libList);

  return 0;
}
