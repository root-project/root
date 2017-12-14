{
// Fill out the code of the actual test
#ifndef SECOND_RUN
   gROOT->ProcessLine(".L test_classes.h+");
#endif

#if defined(ClingWorkAroundMissingDynamicScope) && !defined(SECOND_RUN)
#define SECOND_RUN
   gROOT->ProcessLine(".x runSTL0.C");
#else

   TJsonEx7* ex7 = new TJsonEx7(0);
   TJsonEx8* ex8 = new TJsonEx8(0);
   TJsonEx12* ex12 = new TJsonEx12();
   //ex12->SetValues(0);

   TJsonEx13* ex13 = new TJsonEx13();
   //ex13->SetValues(0);

   cout << " ====== different STL containers TJsonEx7 (len=0) ===== " << endl;
   cout << TBufferJSON::ToJSON(ex7) << endl << endl;
   cout << " ====== STL as basec class TJsonEx8 (len=0) ===== " << endl;
   cout << TBufferJSON::ToJSON(ex8) << endl << endl;
   cout << " ====== STL vector of complex classes TJsonEx12 (len=0) ===== " << endl;
   cout << TBufferJSON::ToJSON(ex12) << endl << endl;
   cout << " ====== More exotic STL containers as set or map with TRef inside TJsonEx13 (len=0) ===== " << endl;
   cout << TBufferJSON::ToJSON(ex13) << endl << endl;

#endif
#ifdef ClingWorkAroundBrokenUnnamedReturn
   gApplication->Terminate(0);
#else
   return 0;
#endif
}
