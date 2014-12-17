#include "libraryLister.h"

void execLoadAllLibs()
{

   gSystem->Setenv("DISPLAY",""); // avoid spurrious warning when loading libGui

   outputRAII o;

   gInterpreter->SetClassAutoparsing(false);

   auto libList = getLibrariesList();
   loadLibrariesInList(libList);

}
