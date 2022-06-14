void P040_TDCacheFile()
{
   gPluginMgr->AddHandler("TFile", "^dcache:", "TDCacheFile",
      "DCache", "TDCacheFile(const char*,Option_t*,const char*,Int_t)");
   gPluginMgr->AddHandler("TFile", "^dcap:", "TDCacheFile",
      "DCache", "TDCacheFile(const char*,Option_t*,const char*,Int_t)");
   gPluginMgr->AddHandler("TFile", "^gsidcap:", "TDCacheFile",
      "DCache", "TDCacheFile(const char*,Option_t*,const char*,Int_t)");
}
