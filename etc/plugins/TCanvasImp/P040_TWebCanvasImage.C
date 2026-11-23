void P040_TWebCanvasImage()
{
   gPluginMgr->AddHandler("TCanvasImp", "TWebCanvasImage", "TWebCanvas",
      "WebGui6", "ProduceImage(TPad *, const char *, Int_t, Int_t)");
}
