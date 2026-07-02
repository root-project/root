void P020_TQt6Canvas()
{
   gPluginMgr->AddHandler("TCanvasImp", "TQt6Canvas", "TQt6Canvas",
      "ROOTQt6Canvas", "NewCanvas(TCanvas *, const char *, Int_t, Int_t, UInt_t, UInt_t)");
}
