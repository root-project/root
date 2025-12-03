#include "test_classes.h"

void runObjects()
{
   gSystem->Load("libJsonTestClasses");

   TJsonEx5 ex5; ex5.Init();
   TJsonEx6 ex6; ex6.Init();
   TJsonEx10 ex10; ex10.Init();
   TString json;

   std::cout << " ====== objects as class members TJsonEx5 ===== " << std::endl;
   json = TBufferJSON::ToJSON(&ex5);
   testJsonReading(json);
   std::cout << json << std::endl << std::endl;
   std::cout << " ====== arrays of objects as class members TJsonEx6 ===== " << std::endl;
   json = TBufferJSON::ToJSON(&ex6);
   testJsonReading(json);
   std::cout << json << std::endl << std::endl;
   std::cout << " ====== ROOT TObject/TNamed/TString as class members TJsonEx10 ===== " << std::endl;
   json = TBufferJSON::ToJSON(&ex10);
   testJsonReading(json);
   std::cout << json << std::endl << std::endl;
   std::cout << " ============ selected data members ======== " << std::endl;
   std::cout << "ex5.fObj1 = " << TBufferJSON::ToJSON(&ex5, 0, "fObj1") << std::endl;
   std::cout << "ex5.fPtr1 = " << TBufferJSON::ToJSON(&ex5, 0, "fPtr1") << std::endl;
   std::cout << "ex5.fSafePtr1 = " << TBufferJSON::ToJSON(&ex5, 0, "fSafePtr1") << std::endl;
   std::cout << "ex6.fObj1 = " << TBufferJSON::ToJSON(&ex6, 0, "fObj1") << std::endl;
}
