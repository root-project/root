void P130_TDavixFile()
{
   TString configfeatures = gROOT->GetConfigFeatures();

   // only if ROOT was compiled with davix enabled do we configure a handler
   if (configfeatures.Contains("davix") &&
       !gEnv->GetValue("Davix.UseOldClient", 0)) {

      gPluginMgr->AddHandler("TFile", "^http[s]?:", "TDavixFile",
      "RDAVIX", "TDavixFile(const char*, Option_t *, const char *,Int_t)");

      gPluginMgr->AddHandler("TFile", "^dav[s]?:", "TDavixFile",
      "RDAVIX", "TDavixFile(const char*, Option_t *, const char *,Int_t)");

      gPluginMgr->AddHandler("TFile", "^s3[s]?:", "TDavixFile",
      "RDAVIX", "TDavixFile(const char*, Option_t *, const char *,Int_t)");

   }
}
