void P130_TDavixFile()
{
   TString configfeatures = gROOT->GetConfigFeatures();

   // only if ROOT was compiled with davix enabled we configure a handler
   if ( configfeatures.Contains("davix") ) {

      gPluginMgr->AddHandler("TFile", "^http[s]?:", "TDavixFile",
      "RDAVIX", "TDavixFile(const char*, Option_t *, const char *,Int_t)");

      gPluginMgr->AddHandler("TFile", "^dav[s]?:", "TDavixFile",
      "RDAVIX", "TDavixFile(const char*, Option_t *, const char *,Int_t)");

      gPluginMgr->AddHandler("TFile", "^s3[s]?:", "TDavixFile",
      "RDAVIX", "TDavixFile(const char*, Option_t *, const char *,Int_t)");

   } else {
      Error("P130_TDavixFile","Please fix your ROOT config to be able to load libdavix.so");
   }
}
