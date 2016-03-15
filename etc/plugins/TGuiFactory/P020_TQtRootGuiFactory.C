TGuiFactory* ROOT_Plugin_TQtRootGuiFactory();

void P020_TQtRootGuiFactory()
{
   gPluginMgr->AddHandler("TGuiFactory", "qt", "TQtRootGuiFactory",
      "QtRoot", "::ROOT_Plugin_TQtRootGuiFactory()");
}
