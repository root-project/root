void P140_TCurlFile()
{
   TString configfeatures = gROOT->GetConfigFeatures();

   if (configfeatures.Contains("curl") &&
       (!configfeatures.Contains("davix") || gEnv->GetValue("Curl.ReplaceDavix", 0))) {

      gPluginMgr->AddHandler("TFile", "^http[s]?:", "TCurlFile", "RCurlHttp", "TCurlFile(const char *, Option_t *)");
   }
}
