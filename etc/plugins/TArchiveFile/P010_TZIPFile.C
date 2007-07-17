void P010_TZIPFile()
{
   gPluginMgr->AddHandler("TArchiveFile", ".+[.]zip$", "TZIPFile",
      "Core", "TZIPFile(const char*,const char*,TFile*)");
}
