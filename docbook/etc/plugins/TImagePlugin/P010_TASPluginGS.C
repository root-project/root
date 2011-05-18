void P010_TASPluginGS()
{
   gPluginMgr->AddHandler("TImagePlugin", "ps", "TASPluginGS",
      "ASPluginGS", "TASPluginGS(const char*)");
   gPluginMgr->AddHandler("TImagePlugin", "eps", "TASPluginGS",
      "ASPluginGS", "TASPluginGS(const char*)");
   gPluginMgr->AddHandler("TImagePlugin", "pdf", "TASPluginGS",
      "ASPluginGS", "TASPluginGS(const char*)");
}
