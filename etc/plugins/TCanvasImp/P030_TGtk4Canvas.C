void P030_TGtk4Canvas()
{
   gPluginMgr->AddHandler("TCanvasImp", "TGtk4Canvas", "ROOT::Experimental::TGtk4Canvas",
      "ROOTGtk4Canvas", "NewCanvas(TCanvas *, const char *, Int_t, Int_t, UInt_t, UInt_t)");
}
