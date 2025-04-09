{
// Fill out the code of the actual test
#ifndef SECOND_RUN
   gROOT->ProcessLine(".L test_classes.h+");
#endif

#if defined(ClingWorkAroundMissingDynamicScope) && !defined(SECOND_RUN)
#define SECOND_RUN
   gROOT->ProcessLine(".x runStreamerLoop.C");
#else

   TJsonEx9 ex9_0;
   TJsonEx9 ex9_1; ex9_1.Init(1);
   TJsonEx9 ex9_7; ex9_7.Init(7);
   TString json;

   cout << " ====== kStreamerLoop members with Counter==0 TJsonEx9 ===== " << endl;
   json = TBufferJSON::ToJSON(&ex9_0);
   testJsonReading(json);
   cout << json << endl << endl;
   cout << " ====== kStreamerLoop members with Counter==1 TJsonEx9 ===== " << endl;
   json = TBufferJSON::ToJSON(&ex9_1);
   testJsonReading(json);
   cout << json << endl << endl;
   cout << " ====== kStreamerLoop members with Counter==7 TJsonEx9 ===== " << endl;
   json = TBufferJSON::ToJSON(&ex9_7);
   testJsonReading(json);
   cout << json << endl << endl;

#endif
#ifdef ClingWorkAroundBrokenUnnamedReturn
   gApplication->Terminate(0);
#else
   return 0;
#endif
}
