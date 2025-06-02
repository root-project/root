void P010_TGeoChecker()
{
   gPluginMgr->AddHandler("TVirtualGeoChecker", "root", "TGeoChecker",
      "GeomChecker", "TGeoChecker(TGeoManager*)");
}
