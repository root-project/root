void P020_TDCacheSystem()
{
   gPluginMgr->AddHandler("TSystem", "^dcache:", "TDCacheSystem",
      "DCache", "TDCacheSystem()");
   gPluginMgr->AddHandler("TSystem", "^dcap:", "TDCacheSystem",
      "DCache", "TDCacheSystem()");
   gPluginMgr->AddHandler("TSystem", "^gsidcap:", "TDCacheSystem",
      "DCache", "TDCacheSystem()");
}
