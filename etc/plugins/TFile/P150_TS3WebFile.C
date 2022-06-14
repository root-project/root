void P150_TS3WebFile()
{
   gPluginMgr->AddHandler("TFile", "^[a]?s3:", "TS3WebFile",
      "Net", "TS3WebFile(const char*,Option_t*)");
   gPluginMgr->AddHandler("TFile", "^s3http[s]?:", "TS3WebFile",
      "Net", "TS3WebFile(const char*,Option_t*)");
   gPluginMgr->AddHandler("TFile", "^gs:", "TS3WebFile",
      "Net", "TS3WebFile(const char*,Option_t*)");
   gPluginMgr->AddHandler("TFile", "^gshttp[s]?:", "TS3WebFile",
      "Net", "TS3WebFile(const char*,Option_t*)");
}
