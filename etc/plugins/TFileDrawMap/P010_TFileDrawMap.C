void P010_TFileDrawMap()
{
   gPluginMgr->AddHandler("TFileDrawMap", "*", "TFileDrawMap",
      "TreePlayer", "TFileDrawMap(const TFile*, const char*, const char*)");
}
