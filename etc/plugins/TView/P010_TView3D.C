void P010_TView3D()
{
   gPluginMgr->AddHandler("TView", "*", "TView3D",
      "Graf3d", "TView3D(Int_t, const Double_t*, const Double_t*)");
}
