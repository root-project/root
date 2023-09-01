void P010_TXSlave()
{
   gPluginMgr->AddHandler("TSlave", "^xpd", "TXSlave",
      "Proofx", "TXSlave(const char *,const char *,Int_t,const char *, TProof *,Int_t,const char *,const char *)");
}
