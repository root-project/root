{
// Fill out the code of the actual test
#ifndef SECOND_RUN
   gROOT->ProcessLine(".L test_classes.h+");
#endif

#if defined(ClingWorkAroundMissingDynamicScope) && !defined(SECOND_RUN)
#define SECOND_RUN
   gROOT->ProcessLine(".x runStreamerLoop.C");
#else

   TJsonEx9* ex9_0 = new TJsonEx9();
   TJsonEx9* ex9_1 = new TJsonEx9();
   ex9_1->SetValues(1);
   TJsonEx9* ex9_7 = new TJsonEx9();
   ex9_7->SetValues(7);

   cout << " ====== kStreamerLoop members with Counter==0 TJsonEx9 ===== " << endl;
   cout << TBufferJSON::ToJSON(ex9_0) << endl << endl;
   cout << " ====== kStreamerLoop members with Counter==1 TJsonEx9 ===== " << endl;
   cout << TBufferJSON::ToJSON(ex9_1) << endl << endl;
   cout << " ====== kStreamerLoop members with Counter==7 TJsonEx9 ===== " << endl;
   cout << TBufferJSON::ToJSON(ex9_7) << endl << endl;

#endif
#ifdef ClingWorkAroundBrokenUnnamedReturn
   gApplication->Terminate(0);
#else
   return 0;
#endif
}
