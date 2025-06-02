void P010_TGeoChecker()
{
   gPluginMgr->AddHandler("TVirtualGeoChecker", "root", "TGeoChecker",
      "GeomPainter", "TGeoChecker(TGeoManager*)");
}
