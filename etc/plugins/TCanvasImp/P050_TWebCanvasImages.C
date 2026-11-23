void P050_TWebCanvasImages()
{
   gPluginMgr->AddHandler("TCanvasImp", "TWebCanvasImages", "TWebCanvas",
      "WebGui6", "ProduceImagesP(std::vector<TPad *> *, const char *, Int_t, Int_t)");
}
