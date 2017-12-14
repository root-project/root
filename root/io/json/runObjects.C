{
// Fill out the code of the actual test
#ifndef SECOND_RUN
   gROOT->ProcessLine(".L test_classes.h+");
#endif

#if defined(ClingWorkAroundMissingDynamicScope) && !defined(SECOND_RUN)
#define SECOND_RUN
   gROOT->ProcessLine(".x runObjects.C");
#else

   TJsonEx5* ex5 = new TJsonEx5(true);
   TJsonEx6* ex6 = new TJsonEx6(true);
   TJsonEx10* ex10 = new TJsonEx10();
   ex10->SetValues();

   cout << " ====== objects as class members TJsonEx5 ===== " << endl;
   cout << TBufferJSON::ToJSON(ex5) << endl << endl;
   cout << " ====== arrays of objects as class members TJsonEx6 ===== " << endl;
   cout << TBufferJSON::ToJSON(ex6) << endl << endl;
   cout << " ====== ROOT TObject/TNamed/TString as class members TJsonEx10 ===== " << endl;
   cout << TBufferJSON::ToJSON(ex10) << endl << endl;
   cout << " ============ selected data members ======== " << endl;
   cout << "ex5.fObj1 = " << TBufferJSON::ConvertToJSON(ex5, gROOT->GetClass("TJsonEx5"), 0, "fObj1") << endl;
   cout << "ex5.fPtr1 = " << TBufferJSON::ConvertToJSON(ex5, gROOT->GetClass("TJsonEx5"), 0, "fPtr1") << endl;
   cout << "ex5.fSafePtr1 = " << TBufferJSON::ConvertToJSON(ex5, gROOT->GetClass("TJsonEx5"), 0, "fSafePtr1") << endl;
   cout << "ex6.fObj1 = " << TBufferJSON::ConvertToJSON(ex6, gROOT->GetClass("TJsonEx6"), 0, "fObj1") << endl;

#endif
#ifdef ClingWorkAroundBrokenUnnamedReturn
   gApplication->Terminate(0);
#else
   return 0;
#endif
}
