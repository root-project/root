void P010_TTreePlayer()
{
   gPluginMgr->AddHandler("TVirtualTreePlayer", "*", "TTreePlayer",
      "TreePlayer", "TTreePlayer()");
}
