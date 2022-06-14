void P040_TImageDump()
{
   gPluginMgr->AddHandler("TVirtualPS", "image", "TImageDump",
      "Postscript", "TImageDump()");
}
