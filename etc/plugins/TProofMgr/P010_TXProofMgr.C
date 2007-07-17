void P010_TXProofMgr()
{
   gPluginMgr->AddHandler("TProofMgr", "^xpd", "TXProofMgr",
      "Proofx", "TXProofMgr(const char *,Int_t,const char *)");
}
