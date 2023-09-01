void P070_TNetSystem()
{
   gPluginMgr->AddHandler("TSystem", "^rootd:", "TNetSystem",
      "Net", "TNetSystem(const char *,Bool_t)");
   gPluginMgr->AddHandler("TSystem", "^rootup:", "TNetSystem",
      "Net", "TNetSystem(const char *,Bool_t)");
   gPluginMgr->AddHandler("TSystem", "^roots:", "TNetSystem",
      "Net", "TNetSystem(const char *,Bool_t)");
   gPluginMgr->AddHandler("TSystem", "^rootk:", "TNetSystem",
      "Net", "TNetSystem(const char *,Bool_t)");
   gPluginMgr->AddHandler("TSystem", "^rootg:", "TNetSystem",
      "Net", "TNetSystem(const char *,Bool_t)");
   gPluginMgr->AddHandler("TSystem", "^rooth:", "TNetSystem",
      "Net", "TNetSystem(const char *,Bool_t)");
   gPluginMgr->AddHandler("TSystem", "^rootug:", "TNetSystem",
      "Net", "TNetSystem(const char *,Bool_t)");
}
