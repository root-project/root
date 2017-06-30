{

   printf("Running rootlogon.C for webgui \n");
   gPluginMgr->AddHandler("TGuiFactory", "web2", "TWebGuiFactory", "WebGui", "TWebGuiFactory()");
   gPluginMgr->AddHandler("TVirtualX", "web2", "TWebVirtualX", "WebGui", "TWebVirtualX(const char*,const char*)");

}
