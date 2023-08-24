void P030_RWebBrowserImp()
{
   gPluginMgr->AddHandler("TBrowserImp", "ROOT::RWebBrowserImp", "ROOT::RWebBrowserImp",
      "ROOTBrowserv7", "NewBrowser(TBrowser *, const char *, Int_t, Int_t, UInt_t, UInt_t, Option_t *)");
   gPluginMgr->AddHandler("TBrowserImp", "ROOT::RWebBrowserImp", "ROOT::RWebBrowserImp",
      "ROOTBrowserv7", "NewBrowser(TBrowser *, const char *, UInt_t, UInt_t, Option_t *)");
}
