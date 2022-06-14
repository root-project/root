void P030_TProofLite()
{
   gPluginMgr->AddHandler("TProof", "^lite:", "TProofLite",
      "Proof", "TProofLite(const char*,const char*,const char*,Int_t,const char*,TProofMgr *)");
}
