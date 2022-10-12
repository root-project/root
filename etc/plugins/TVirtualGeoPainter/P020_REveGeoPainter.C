void P020_RGeoPainter()
{
   gPluginMgr->AddHandler("TVirtualGeoPainter", "web", "ROOT::Experimental::RGeoPainter",
      "ROOTGeomViewer", "RGeoPainter(TGeoManager*)");
}
