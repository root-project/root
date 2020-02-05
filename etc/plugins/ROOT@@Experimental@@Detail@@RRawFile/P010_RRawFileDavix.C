void P010_RRawFileDavix()
{
   TString configfeatures = gROOT->GetConfigFeatures();

   // only if ROOT was compiled with davix enabled do we configure a handler
   if (configfeatures.Contains("davix") &&
       !gEnv->GetValue("Davix.UseOldClient", 0)) {

      gPluginMgr->AddHandler(
         "ROOT::Experimental::Detail::RRawFile",
         "^http[s]?:",
         "ROOT::Experimental::Detail::RRawFileDavix",
         "RDAVIX",
         "RRawFileDavix(std::string_view, ROOT::Experimental::Detail::RRawFile::ROptions)");
   }
}
