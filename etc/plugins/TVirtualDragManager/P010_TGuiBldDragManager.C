void P010_TGuiBldDragManager()
{
   gPluginMgr->AddHandler("TVirtualDragManager", "*", "TGuiBldDragManager",
      "GuiBld", "TGuiBldDragManager()");
}
