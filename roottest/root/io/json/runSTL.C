#include "test_classes.h"

void runSTL()
{
   gSystem->Load("libJsonTestClasses");

   TJsonEx7 ex7; ex7.Init();
   TJsonEx8 ex8; ex8.Init();
   TJsonEx12 ex12; ex12.Init();
   TJsonEx13 ex13; ex13.Init();
   TString json;

   std::cout << " ====== different STL containers TJsonEx7 ===== " << std::endl;
   json = TBufferJSON::ToJSON(&ex7);
   testJsonReading(json);
   std::cout << json << std::endl << std::endl;
   std::cout << " ====== STL as base class TJsonEx8 ===== " << std::endl;
   json = TBufferJSON::ToJSON(&ex8);
   testJsonReading(json);
   std::cout << json << std::endl << std::endl;
   std::cout << " ====== STL vector of complex classes TJsonEx12 ===== " << std::endl;
   json = TBufferJSON::ToJSON(&ex12);
   testJsonReading(json);
   std::cout << json << std::endl << std::endl;
   std::cout << " ====== More exotic STL containers as set or map with TRef inside TJsonEx13 ===== " << std::endl;
   json = TBufferJSON::ToJSON(&ex13);
   testJsonReading(json);
   std::cout << json << std::endl << std::endl;
}
