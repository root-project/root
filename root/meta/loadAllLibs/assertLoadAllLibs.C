#include "libraryLister.h"

int assertLoadAllLibs()
{

   gSystem->Setenv("DISPLAY",""); // avoid spurrious warning when loading libGui

   //outputRAII o;

   gInterpreter->SetClassAutoparsing(false);

   auto libList = getLibrariesList();
   libList.unique();
   loadLibrariesInList(libList);
 
   return 0;
}
