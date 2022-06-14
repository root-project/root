void P020_TRootBrowserLite()
{
   gPluginMgr->AddHandler("TBrowserImp", "TRootBrowserLite", "TRootBrowserLite",
      "Gui", "NewBrowser(TBrowser *, const char *, Int_t, Int_t, UInt_t, UInt_t)");
   gPluginMgr->AddHandler("TBrowserImp", "TRootBrowserLite", "TRootBrowserLite",
      "Gui", "NewBrowser(TBrowser *, const char *, UInt_t, UInt_t)");
}
