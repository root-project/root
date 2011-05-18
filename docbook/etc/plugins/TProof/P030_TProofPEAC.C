void P030_TProofPEAC()
{
   gPluginMgr->AddHandler("TProof", "^peac:", "TProofPEAC",
      "Peac", "TProofPEAC(const char*,const char*,const char*,Int_t,const char*)");
}
