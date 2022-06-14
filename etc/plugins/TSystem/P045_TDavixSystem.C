void P045_TDavixSystem()
{
   TString configfeatures = gROOT->GetConfigFeatures();

   // only if ROOT was compiled with davix enabled do we configure a handler
   if (configfeatures.Contains("davix") &&
       !gEnv->GetValue("Davix.UseOldClient", 0)) {

      gPluginMgr->AddHandler("TSystem", "^http[s]?:", "TDavixSystem",
         "RDAVIX", "TDavixSystem()");

      gPluginMgr->AddHandler("TSystem", "^dav[s]?:", "TDavixSystem",
         "RDAVIX", "TDavixSystem()");

      gPluginMgr->AddHandler("TSystem", "^s3[s]?:", "TDavixSystem",
         "RDAVIX", "TDavixSystem()");

   }
}
