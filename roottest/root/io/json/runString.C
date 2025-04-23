#include "test_classes.h"

void runString()
{
   gSystem->Load("libJsonTestClasses");

   TJsonEx4 ex4; ex4.Init();

   std::cout << " ====== string data types TJsonEx4 ===== " << std::endl;
   TString json = TBufferJSON::ToJSON(&ex4);
   testJsonReading(json);
   std::cout << json << std::endl << std::endl;
   std::cout << " ============ selected data members ======== " << std::endl;
   std::cout << "ex4.fStr1 = " << TBufferJSON::ToJSON(&ex4, 0, "fStr1") << std::endl;
   std::cout << "ex4.fStr2 = " << TBufferJSON::ToJSON(&ex4, 0, "fStr2") << std::endl;
   std::cout << "ex4.fStr3 = " << TBufferJSON::ToJSON(&ex4, 0, "fStr3") << std::endl;
   std::cout << "ex4.fStr4 = " << TBufferJSON::ToJSON(&ex4, 0, "fStr4") << std::endl;
}
