{
   new TFile("digi.root");
#ifdef ClingWorkAroundMissingDynamicScope
   TTree *Digi; gFile->GetObject("Digi",Digi);
#endif
#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
   int n;
   n = Digi->BuildIndex("m_runId", "m_eventId");
#else
int n = Digi->BuildIndex("m_runId", "m_eventId");
#endif
#ifdef ClingWorkAroundBrokenUnnamedReturn
   gApplication->Terminate(!(n>0)); // to signal succesfull we need a zero!
#else
   return ! (n>0); // to signal succesfull we need a zero!
#endif
}
