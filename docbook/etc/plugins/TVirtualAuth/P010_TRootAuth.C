void P010_TRootAuth()
{
   gPluginMgr->AddHandler("TVirtualAuth", "Root", "TRootAuth",
      "RootAuth", "TRootAuth()");
}
