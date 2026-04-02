void P120_TNetFile()
{
   gPluginMgr->AddHandler("TFile", "^rootd:", "ROOT::Deprecated::TNetFile",
      "Net", "ROOT::Deprecated::TNetFile(const char*,Option_t*,const char*,Int_t,Int_t)");
   gPluginMgr->AddHandler("TFile", "^rootup:", "ROOT::Deprecated::TNetFile",
      "Net", "ROOT::Deprecated::TNetFile(const char*,Option_t*,const char*,Int_t,Int_t)");
   gPluginMgr->AddHandler("TFile", "^roots:", "ROOT::Deprecated::TNetFile",
      "Net", "ROOT::Deprecated::TNetFile(const char*,Option_t*,const char*,Int_t,Int_t)");
   gPluginMgr->AddHandler("TFile", "^rootk:", "ROOT::Deprecated::TNetFile",
      "Net", "ROOT::Deprecated::TNetFile(const char*,Option_t*,const char*,Int_t,Int_t)");
   gPluginMgr->AddHandler("TFile", "^rootg:", "ROOT::Deprecated::TNetFile",
      "Net", "ROOT::Deprecated::TNetFile(const char*,Option_t*,const char*,Int_t,Int_t)");
   gPluginMgr->AddHandler("TFile", "^rooth:", "ROOT::Deprecated::TNetFile",
      "Net", "ROOT::Deprecated::TNetFile(const char*,Option_t*,const char*,Int_t,Int_t)");
   gPluginMgr->AddHandler("TFile", "^rootug:", "ROOT::Deprecated::TNetFile",
      "Net", "ROOT::Deprecated::TNetFile(const char*,Option_t*,const char*,Int_t,Int_t)");
}
