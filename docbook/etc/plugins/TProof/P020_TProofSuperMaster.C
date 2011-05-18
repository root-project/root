void P020_TProofSuperMaster()
{
   gPluginMgr->AddHandler("TProof", "^sm:", "TProofSuperMaster",
      "Proof", "TProofSuperMaster(const char*,const char*,const char*,Int_t,const char*)");
}
