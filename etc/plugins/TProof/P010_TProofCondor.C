void P010_TProofCondor()
{
   gPluginMgr->AddHandler("TProof", "^condor:", "TProofCondor",
      "Proof", "TProofCondor(const char*,const char*,const char*,Int_t,const char*)");
}
