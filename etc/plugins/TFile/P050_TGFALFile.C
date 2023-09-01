void P050_TGFALFile()
{
   gPluginMgr->AddHandler("TFile", "^gfal:", "TGFALFile",
      "GFAL", "TGFALFile(const char*,Option_t*,const char*,Int_t)");
}
