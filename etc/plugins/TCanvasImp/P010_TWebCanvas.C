void P010_TWebCanvas()
{
   gPluginMgr->AddHandler("TCanvasImp", "TWebCanvas", "TWebCanvas",
      "WebGui6", "NewCanvas(TCanvas *, const char *, Int_t, Int_t, UInt_t, UInt_t)");
}
