#include "test_classes.h"

void runSTL1()
{
   gSystem->Load("libJsonTestClasses");

   TJsonEx7 ex7; ex7.Init(1);
   TJsonEx8 ex8; ex8.Init(1);
   TJsonEx12 ex12; ex12.Init(1);
   TJsonEx13 ex13; ex13.Init(1);
   TString json;

   std::cout << " ====== different STL containers TJsonEx7 (len=1) ===== " << std::endl;
   json = TBufferJSON::ToJSON(&ex7);
   testJsonReading(json);
   std::cout << json << std::endl << std::endl;
   std::cout << " ====== STL as base class TJsonEx8 (len=1) ===== " << std::endl;
   json = TBufferJSON::ToJSON(&ex8);
   testJsonReading(json);
   std::cout << json << std::endl << std::endl;
   std::cout << " ====== STL vector of complex classes TJsonEx12 (len=1) ===== " << std::endl;
   json = TBufferJSON::ToJSON(&ex12);
   testJsonReading(json);
   std::cout << json << std::endl << std::endl;
   std::cout << " ====== More exotic STL containers as set or map with TRef inside TJsonEx13 (len=1) ===== " << std::endl;
   json = TBufferJSON::ToJSON(&ex13);
   testJsonReading(json);
   std::cout << json << std::endl << std::endl;
}
