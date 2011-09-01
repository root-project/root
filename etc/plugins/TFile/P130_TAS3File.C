void P130_TAS3File()
{
   gPluginMgr->AddHandler("TFile", "^as3:", "TAS3File",
      "Net", "TAS3File(const char*,Option_t*)");
}
