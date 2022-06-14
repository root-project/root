void P050_TWebSystem()
{
   gPluginMgr->AddHandler("TSystem", "^http:", "TWebSystem",
      "Net", "TWebSystem()");
}
