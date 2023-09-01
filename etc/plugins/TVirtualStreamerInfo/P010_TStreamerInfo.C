void P010_TStreamerInfo()
{
   gPluginMgr->AddHandler("TVirtualStreamerInfo", "*", "TStreamerInfo",
      "RIO", "TStreamerInfo()");
}
