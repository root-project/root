void P090_TSQLFile()
{
   gPluginMgr->AddHandler("TFile", "^mysql:", "TSQLFile",
      "SQLIO", "TSQLFile(const char*,Option_t*,const char*,const char*)");
   gPluginMgr->AddHandler("TFile", "^oracle:", "TSQLFile",
      "SQLIO", "TSQLFile(const char*,Option_t*,const char*,const char*)");
}
