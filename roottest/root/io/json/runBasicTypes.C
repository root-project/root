#include "test_classes.h"

void runBasicTypes()
{
   gSystem->Load("libJsonTestClasses");

   TJsonEx1 ex1; ex1.Init();
   TJsonEx2 ex2; ex2.Init();
   TJsonEx3 ex3; ex3.Init();
   TJsonEx11 ex11; ex11.Init();
   TString json;

   std::cout << " ====== basic data types TJsonEx1 ===== " << std::endl;
   json = TBufferJSON::ToJSON(&ex1);
   testJsonReading(json);
   std::cout << json << std::endl << std::endl;
   std::cout << " ====== inheritance from TJsonEx1, TJsonEx11 ===== " << std::endl;
   json = TBufferJSON::ToJSON(&ex11);
   testJsonReading(json);
   std::cout << json << std::endl << std::endl;
   std::cout << " ====== arrays of basic data types TJsonEx2 ===== " << std::endl;
   json = TBufferJSON::ToJSON(&ex2);
   testJsonReading(json);
   std::cout << json << std::endl << std::endl;
   std::cout << " ====== dynamic arrays of basic data types TJsonEx3 ===== " << std::endl;
   json = TBufferJSON::ToJSON(&ex3);
   testJsonReading(json);
   std::cout << json << std::endl << std::endl;
   std::cout << " ============ selected data members ======== " << std::endl;
   std::cout << "ex2.fInt = " << TBufferJSON::ToJSON(&ex2, 0, "fInt") << std::endl;
   std::cout << "ex3.fChar = " << TBufferJSON::ToJSON(&ex3, 0, "fChar") << std::endl;
   std::cout << "ex3.fLong = " << TBufferJSON::ToJSON(&ex3, 0, "fLong") << std::endl;
}
