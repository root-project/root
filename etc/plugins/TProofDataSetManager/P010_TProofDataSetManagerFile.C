void P010_TProofDataSetManagerFile()
{
   gPluginMgr->AddHandler("TProofDataSetManager", "^file", "TProofDataSetManagerFile", "Proof",
                          "TProofDataSetManagerFile(const char *,const char *,const char *)");
}
