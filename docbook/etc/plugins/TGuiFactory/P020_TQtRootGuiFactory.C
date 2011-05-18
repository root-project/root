void P020_TQtRootGuiFactory()
{
   gPluginMgr->AddHandler("TGuiFactory", "qt", "TQtRootGuiFactory",
      "QtRoot", "TQtRootGuiFactory()");
}
