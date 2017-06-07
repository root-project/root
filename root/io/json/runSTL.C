{
// Fill out the code of the actual test
#ifndef SECOND_RUN
   gROOT->ProcessLine(".L test_classes.h+");
#endif

#if defined(ClingWorkAroundMissingDynamicScope) && !defined(SECOND_RUN)
#define SECOND_RUN
   gROOT->ProcessLine(".x runSTL.C");
#else

   TJsonEx7* ex7 = new TJsonEx7(10);
   TJsonEx8* ex8 = new TJsonEx8(10);
   TJsonEx12* ex12 = new TJsonEx12();
   ex12->SetValues(10);

   TJsonEx13* ex13 = new TJsonEx13();
   ex13->SetValues(10);


   cout << " ====== different STL containers TJsonEx7 ===== " << endl;
   cout << TBufferJSON::ConvertToJSON(ex7, gROOT->GetClass("TJsonEx7")) << endl << endl;
   cout << " ====== STL as basec class TJsonEx8 ===== " << endl;
   cout << TBufferJSON::ConvertToJSON(ex8, gROOT->GetClass("TJsonEx8")) << endl << endl;
   cout << " ====== STL vector of complex classes TJsonEx12 ===== " << endl;
   cout << TBufferJSON::ConvertToJSON(ex12, gROOT->GetClass("TJsonEx12")) << endl << endl;
   cout << " ====== More exotic STL containers as set or map with TRef inside TJsonEx13 ===== " << endl;
   cout << TBufferJSON::ConvertToJSON(ex13, gROOT->GetClass("TJsonEx13")) << endl << endl;

#endif
#ifdef ClingWorkAroundBrokenUnnamedReturn
   gApplication->Terminate(0);
#else
   return 0;
#endif
}
