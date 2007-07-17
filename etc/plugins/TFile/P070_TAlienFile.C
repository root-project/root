void P070_TAlienFile()
{
   gPluginMgr->AddHandler("TFile", "^alien:", "TAlienFile",
      "RAliEn", "Open(const char*,Option_t*,const char*,Int_t,Bool_t)");
}
