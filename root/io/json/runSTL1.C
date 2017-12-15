{
// Fill out the code of the actual test
#ifndef SECOND_RUN
   gROOT->ProcessLine(".L test_classes.h+");
#endif

#if defined(ClingWorkAroundMissingDynamicScope) && !defined(SECOND_RUN)
#define SECOND_RUN
   gROOT->ProcessLine(".x runSTL1.C");
#else

   TJsonEx7 *ex7 = new TJsonEx7; ex7->Init(1);
   TJsonEx8 *ex8 = new TJsonEx8; ex8->Init(1);
   TJsonEx12 *ex12 = new TJsonEx12; ex12->Init(1);
   TJsonEx13 *ex13 = new TJsonEx13; ex13->Init(1);
   TString json;

   cout << " ====== different STL containers TJsonEx7 (len=1) ===== " << endl;
   json = TBufferJSON::ToJSON(ex7);
   testJsonReading(json);
   cout << json << endl << endl;
   cout << " ====== STL as base class TJsonEx8 (len=1) ===== " << endl;
   json = TBufferJSON::ToJSON(ex8);
   testJsonReading(json);
   cout << json << endl << endl;
   cout << " ====== STL vector of complex classes TJsonEx12 (len=1) ===== " << endl;
   json = TBufferJSON::ToJSON(ex12);
   testJsonReading(json);
   cout << json << endl << endl;
   cout << " ====== More exotic STL containers as set or map with TRef inside TJsonEx13 (len=1) ===== " << endl;
   json = TBufferJSON::ToJSON(ex13);
   testJsonReading(json);
   cout << json << endl << endl;

   delete ex7;
   delete ex8;
   delete ex12;
   delete ex13;

#endif
#ifdef ClingWorkAroundBrokenUnnamedReturn
   gApplication->Terminate(0);
#else
   return 0;
#endif
}
