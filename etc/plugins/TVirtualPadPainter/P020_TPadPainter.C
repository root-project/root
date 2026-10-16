void P020_TPadPainter()
{
   gPluginMgr->AddHandler("TVirtualPadPainter", "native", "TPadPainter",
                          "ROOTPadPainter", "TPadPainter()");
}
