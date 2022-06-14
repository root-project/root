void P020_TDataSetManagerAliEn()
{
   gPluginMgr->AddHandler("TDataSetManager", "^alien", "TDataSetManagerAliEn", "Proof",
                          "TDataSetManagerAliEn(const char *,const char *,const char *)");
}
