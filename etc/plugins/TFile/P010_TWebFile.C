void P010_TWebFile()
{
   gPluginMgr->AddHandler("TFile", "^http[s]?:", "ROOT::Deprecated::TWebFile",
      "Net", "ROOT::Deprecated::TWebFile(const char*,Option_t*)");
}
