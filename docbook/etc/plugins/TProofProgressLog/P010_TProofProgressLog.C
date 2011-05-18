void P010_TProofProgressLog()
{
   gPluginMgr->AddHandler("TProofProgressLog", "*", "TProofProgressLog",
      "SessionViewer", "TProofProgressLog(const char*,Int_t,Long64_t,Long64_t)");
}
