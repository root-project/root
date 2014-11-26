#include "libraryLister.h"

void execLoadAllLibs()
{

   outputRAII o;

   gInterpreter->SetClassAutoparsing(false);

   auto libList = getLibrariesList();
   loadLibrariesInList(libList);

}
