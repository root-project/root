void P150_TS3WebFile()
{
   gPluginMgr->AddHandler("TFile", "^[a]?s3:", "ROOT::Deprecated::TS3WebFile",
      "Net", "ROOT::Deprecated::TS3WebFile(const char*,Option_t*)");
   gPluginMgr->AddHandler("TFile", "^s3http[s]?:", "ROOT::Deprecated::TS3WebFile",
      "Net", "ROOT::Deprecated::TS3WebFile(const char*,Option_t*)");
   gPluginMgr->AddHandler("TFile", "^gs:", "ROOT::Deprecated::TS3WebFile",
      "Net", "ROOT::Deprecated::TS3WebFile(const char*,Option_t*)");
   gPluginMgr->AddHandler("TFile", "^gshttp[s]?:", "ROOT::Deprecated::TS3WebFile",
      "Net", "ROOT::Deprecated::TS3WebFile(const char*,Option_t*)");
}
