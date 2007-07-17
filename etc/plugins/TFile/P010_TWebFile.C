void P010_TWebFile()
{
   gPluginMgr->AddHandler("TFile", "^http:", "TWebFile",
      "Net", "TWebFile(const char*)");
}
