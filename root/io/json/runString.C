{
// Fill out the code of the actual test
#ifndef SECOND_RUN
   gROOT->ProcessLine(".L test_classes.h+");
#endif

#if defined(ClingWorkAroundMissingDynamicScope) && !defined(SECOND_RUN)
#define SECOND_RUN
   gROOT->ProcessLine(".x runString.C");
#else

   TJsonEx4* ex4 = new TJsonEx4(true);

   cout << " ====== string data types TJsonEx4 ===== " << endl;
   cout << TBufferJSON::ToJSON(ex4) << endl << endl;
   cout << " ============ selected data members ======== " << endl;
   cout << "ex4.fStr1 = " << TBufferJSON::ToJSON(ex4, 0, "fStr1") << endl;
   cout << "ex4.fStr2 = " << TBufferJSON::ToJSON(ex4, 0, "fStr2") << endl;
   cout << "ex4.fStr3 = " << TBufferJSON::ToJSON(ex4, 0, "fStr3") << endl;
   cout << "ex4.fStr4 = " << TBufferJSON::ToJSON(ex4, 0, "fStr4") << endl;

#endif
#ifdef ClingWorkAroundBrokenUnnamedReturn
   gApplication->Terminate(0);
#else
   return 0;
#endif
}
