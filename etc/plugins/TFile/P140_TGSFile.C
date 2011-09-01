void P140_TGSFile()
{
   gPluginMgr->AddHandler("TFile", "^gs:", "TGSFile",
      "Net", "TGSFile(const char*,Option_t*)");
}
