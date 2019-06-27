void P010_TASPluginGS()
{
   gPluginMgr->AddHandler("TImagePlugin", "ps", "TASPluginGS",
      "ASImage", "TASPluginGS(const char*)");
   gPluginMgr->AddHandler("TImagePlugin", "eps", "TASPluginGS",
      "ASImage", "TASPluginGS(const char*)");
   gPluginMgr->AddHandler("TImagePlugin", "pdf", "TASPluginGS",
      "ASImage", "TASPluginGS(const char*)");
}
