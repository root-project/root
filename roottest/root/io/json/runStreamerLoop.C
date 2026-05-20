#include "test_classes.h"

void runStreamerLoop()
{
   gSystem->Load("libJsonTestClasses");

   TJsonEx9 ex9_0;
   TJsonEx9 ex9_1; ex9_1.Init(1);
   TJsonEx9 ex9_7; ex9_7.Init(7);
   TString json;

   std::cout << " ====== kStreamerLoop members with Counter==0 TJsonEx9 ===== " << std::endl;
   json = TBufferJSON::ToJSON(&ex9_0);
   testJsonReading(json);
   std::cout << json << std::endl << std::endl;
   std::cout << " ====== kStreamerLoop members with Counter==1 TJsonEx9 ===== " << std::endl;
   json = TBufferJSON::ToJSON(&ex9_1);
   testJsonReading(json);
   std::cout << json << std::endl << std::endl;
   std::cout << " ====== kStreamerLoop members with Counter==7 TJsonEx9 ===== " << std::endl;
   json = TBufferJSON::ToJSON(&ex9_7);
   testJsonReading(json);
   std::cout << json << std::endl << std::endl;
}
