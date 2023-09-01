void P010_TGeoManagerEditor()
{
   gPluginMgr->AddHandler("TGeoManagerEditor", "*", "TGeoManagerEditor",
      "GeomBuilder", "LoadLib()");
}
