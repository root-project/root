void P020_TProofMonSenderSQL()
{
   gPluginMgr->AddHandler("TProofMonSender", "SQL", "TProofMonSenderSQL",
      "ProofPlayer", "TProofMonSenderSQL(const char *,const char *,const char *,const char *,const char *,const char *)");
}
