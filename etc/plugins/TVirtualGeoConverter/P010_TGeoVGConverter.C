void P010_TGeoVGConverter()
{
   gPluginMgr->AddHandler("TVirtualGeoConverter", "*", "TGeoVGConverter",
      "ConverterVG", "TGeoVGConverter(TGeoManager*)");
}
