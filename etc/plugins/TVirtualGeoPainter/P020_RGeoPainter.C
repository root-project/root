void P020_RGeoPainter()
{
   gPluginMgr->AddHandler("TVirtualGeoPainter", "web", "ROOT::RGeoPainter",
      "ROOTGeomViewer", "RGeoPainter(TGeoManager*)");
}
