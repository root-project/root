void P045_TDavixSystem()
{
   TString configfeatures = gROOT->GetConfigFeatures();

   // only if ROOT was compiled with davix enabled we configure a handler
   if ( configfeatures.Contains("davix") ) {

      gPluginMgr->AddHandler("TSystem", "^http[s]?:", "TDavixSystem",
         "RDAVIX", "TDavixSystem()");

      gPluginMgr->AddHandler("TSystem", "^dav[s]?:", "TDavixSystem",
         "RDAVIX", "TDavixSystem()");

      gPluginMgr->AddHandler("TSystem", "^s3[s]?:", "TDavixSystem",
         "RDAVIX", "TDavixSystem()");

   } else {
      Error("P045_TDavixSystem","Please fix your ROOT config to be able to load libdavix.so");
   }
}
