{
// Fill out the code of the actual test
#ifndef SECOND_RUN
   gROOT->ProcessLine(".L test_classes.h+");
#endif

#if defined(ClingWorkAroundMissingDynamicScope) && !defined(SECOND_RUN)
#define SECOND_RUN
   gROOT->ProcessLine(".x runObjects.C");
#else

   TJsonEx5 *ex5 = new TJsonEx5(); ex5->Init();
   TJsonEx6 *ex6 = new TJsonEx6(); ex6->Init();
   TJsonEx10 *ex10 = new TJsonEx10(); ex10->Init();
   TString json;

   cout << " ====== objects as class members TJsonEx5 ===== " << endl;
   json = TBufferJSON::ToJSON(ex5);
   testJsonReading(json);
   cout << json << endl << endl;
   cout << " ====== arrays of objects as class members TJsonEx6 ===== " << endl;
   json = TBufferJSON::ToJSON(ex6);
   testJsonReading(json);
   cout << json << endl << endl;
   cout << " ====== ROOT TObject/TNamed/TString as class members TJsonEx10 ===== " << endl;
   json = TBufferJSON::ToJSON(ex10);
   testJsonReading(json);
   cout << json << endl << endl;
   cout << " ============ selected data members ======== " << endl;
   cout << "ex5.fObj1 = " << TBufferJSON::ToJSON(ex5, 0, "fObj1") << endl;
   cout << "ex5.fPtr1 = " << TBufferJSON::ToJSON(ex5, 0, "fPtr1") << endl;
   cout << "ex5.fSafePtr1 = " << TBufferJSON::ToJSON(ex5, 0, "fSafePtr1") << endl;
   cout << "ex6.fObj1 = " << TBufferJSON::ToJSON(ex6, 0, "fObj1") << endl;

   delete ex5;
   delete ex6;
   delete ex10;

#endif
#ifdef ClingWorkAroundBrokenUnnamedReturn
   gApplication->Terminate(0);
#else
   return 0;
#endif
}
