void P010_TGeoVGConverter()
{
   gPluginMgr->AddHandler("TVirtualGeoConverter", "*", "TGeoVGConverter",
      "VecGeom", "TGeoVGConverter(TGeoManager*)");
}
