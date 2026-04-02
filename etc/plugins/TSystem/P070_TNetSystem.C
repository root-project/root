void P070_TNetSystem()
{
   gPluginMgr->AddHandler("TSystem", "^rootd:", "ROOT::Deprecated::TNetSystem",
      "Net", "ROOT::Deprecated::TNetSystem(const char *,Bool_t)");
   gPluginMgr->AddHandler("TSystem", "^rootup:", "ROOT::Deprecated::TNetSystem",
      "Net", "ROOT::Deprecated::TNetSystem(const char *,Bool_t)");
   gPluginMgr->AddHandler("TSystem", "^roots:", "ROOT::Deprecated::TNetSystem",
      "Net", "ROOT::Deprecated::TNetSystem(const char *,Bool_t)");
   gPluginMgr->AddHandler("TSystem", "^rootk:", "ROOT::Deprecated::TNetSystem",
      "Net", "ROOT::Deprecated::TNetSystem(const char *,Bool_t)");
   gPluginMgr->AddHandler("TSystem", "^rootg:", "ROOT::Deprecated::TNetSystem",
      "Net", "ROOT::Deprecated::TNetSystem(const char *,Bool_t)");
   gPluginMgr->AddHandler("TSystem", "^rooth:", "ROOT::Deprecated::TNetSystem",
      "Net", "ROOT::Deprecated::TNetSystem(const char *,Bool_t)");
   gPluginMgr->AddHandler("TSystem", "^rootug:", "ROOT::Deprecated::TNetSystem",
      "Net", "ROOT::Deprecated::TNetSystem(const char *,Bool_t)");
}
