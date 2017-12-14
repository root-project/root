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
   cout << TBufferJSON::ToJSON(ex7) << endl << endl;
   cout << " ====== STL as basec class TJsonEx8 ===== " << endl;
   cout << TBufferJSON::ToJSON(ex8) << endl << endl;
   cout << " ====== STL vector of complex classes TJsonEx12 ===== " << endl;
   cout << TBufferJSON::ToJSON(ex12) << endl << endl;
   cout << " ====== More exotic STL containers as set or map with TRef inside TJsonEx13 ===== " << endl;
   cout << TBufferJSON::ToJSON(ex13) << endl << endl;

#endif
#ifdef ClingWorkAroundBrokenUnnamedReturn
   gApplication->Terminate(0);
#else
   return 0;
#endif
}
