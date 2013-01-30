void P010_TWebFile()
{
   gPluginMgr->AddHandler("TFile", "^http[s]?:", "TWebFile",
      "Net", "TWebFile(const char*,Option_t*)");
}
