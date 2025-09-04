void P015_RRawFileCurl()
{
   TString configfeatures = gROOT->GetConfigFeatures();

   if (configfeatures.Contains("curl") &&
       (!configfeatures.Contains("davix") || gEnv->GetValue("Curl.ReplaceDavix", 0))) {

      gPluginMgr->AddHandler("ROOT::Internal::RRawFile", "^http[s]?:", "ROOT::Internal::RRawFileCurl", "RCurlHttp",
                             "RRawFileCurl(std::string_view, ROOT::Internal::RRawFile::ROptions)");
   }
}
