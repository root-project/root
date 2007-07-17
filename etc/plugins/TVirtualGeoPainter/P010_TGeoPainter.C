void P010_TGeoPainter()
{
   gPluginMgr->AddHandler("TVirtualGeoPainter", "*", "TGeoPainter",
      "GeomPainter", "TGeoPainter(TGeoManager*)");
}
