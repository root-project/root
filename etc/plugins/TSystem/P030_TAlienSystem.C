void P030_TAlienSystem()
{
   gPluginMgr->AddHandler("TSystem", "^alien:", "TAlienSystem",
      "RAliEn", "TAlienSystem()");
}
