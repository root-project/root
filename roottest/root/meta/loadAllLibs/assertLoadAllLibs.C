#include "libraryLister.h"

int assertLoadAllLibs()
{

   gSystem->Setenv("DISPLAY",""); // avoid spurrious warning when loading libGui

   //outputRAII o;

   gInterpreter->SetClassAutoparsing(false);

   auto libList = getLibrariesList();
   libList.unique();
   //libList.remove_if([](std::string){ static int n = 0; return (n++ % 2 == 0);});
   loadLibrariesInList(libList);
 
   return 0;
}
