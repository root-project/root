void P010_TXProofServ()
{
   gPluginMgr->AddHandler("TProofServ", "^xpd", "TXProofServ",
      "Proofx", "TXProofServ(Int_t *,char **)");
}
