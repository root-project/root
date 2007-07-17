void P020_TRFIOFile()
{
   gPluginMgr->AddHandler("TFile", "^rfio:", "TRFIOFile",
      "RFIO", "TRFIOFile(const char*,Option_t*,const char*,Int_t)");
}
