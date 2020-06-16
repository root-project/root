#include "utils.h"

int execAtlasTest(){

   std::cout << "Checking normalisation of names:\n";

   // As a warm-up
   printNames("std::string");
   printNames("std::vector<std::string>");

   // Load the atlas lib and check some names
   if (0!=gSystem->Load("libAtlasTest_dictrflx"))
      std::cerr << "Error loading dictionary library.\n";

#ifdef ClingWorkAroundAutoParseRecurse
   gInterpreter->AutoParse("Atlas::ClassA");
#endif

   printNames("Atlas::ClassA<Atlas::ClassB, float>");
   printNames("Atlas::ClassA<Atlas::ClassC, int>");

return 0;
}
