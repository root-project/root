void P010_TGX11()
{
   gPluginMgr->AddHandler("TVirtualX", "x11", "TGX11",
      "GX11", "TGX11(const char*,const char*)");
}
