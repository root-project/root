void P010_TRootBrowser()
{
   gPluginMgr->AddHandler("TBrowserImp", "TRootBrowser", "TRootBrowser",
      "Gui", "NewBrowser(TBrowser *, const char *, Int_t, Int_t, UInt_t, UInt_t)");
   gPluginMgr->AddHandler("TBrowserImp", "TRootBrowser", "TRootBrowser",
      "Gui", "NewBrowser(TBrowser *, const char *, UInt_t, UInt_t)");
}
