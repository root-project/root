void P120_TNetFile()
{
   gPluginMgr->AddHandler("TFile", "^rootd:", "TNetFile",
      "Net", "TNetFile(const char*,Option_t*,const char*,Int_t,Int_t)");
   gPluginMgr->AddHandler("TFile", "^rootup:", "TNetFile",
      "Net", "TNetFile(const char*,Option_t*,const char*,Int_t,Int_t)");
   gPluginMgr->AddHandler("TFile", "^roots:", "TNetFile",
      "Net", "TNetFile(const char*,Option_t*,const char*,Int_t,Int_t)");
   gPluginMgr->AddHandler("TFile", "^rootk:", "TNetFile",
      "Net", "TNetFile(const char*,Option_t*,const char*,Int_t,Int_t)");
   gPluginMgr->AddHandler("TFile", "^rootg:", "TNetFile",
      "Net", "TNetFile(const char*,Option_t*,const char*,Int_t,Int_t)");
   gPluginMgr->AddHandler("TFile", "^rooth:", "TNetFile",
      "Net", "TNetFile(const char*,Option_t*,const char*,Int_t,Int_t)");
   gPluginMgr->AddHandler("TFile", "^rootug:", "TNetFile",
      "Net", "TNetFile(const char*,Option_t*,const char*,Int_t,Int_t)");
}
