void P010_TDataSetManagerFile()
{
   gPluginMgr->AddHandler("TDataSetManager", "^file", "TDataSetManagerFile", "Proof",
                          "TDataSetManagerFile(const char *,const char *,const char *)");
}
