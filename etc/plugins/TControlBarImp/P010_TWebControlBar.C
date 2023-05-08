void P010_TWebControlBar()
{
   gPluginMgr->AddHandler("TControlBarImp", "TWebControlBar", "TWebControlBar",
      "WebGui6", "NewControlBar(TControlBar *, const char *, Int_t, Int_t)");
}
