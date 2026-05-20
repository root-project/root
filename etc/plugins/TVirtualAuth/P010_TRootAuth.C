void P010_TRootAuth()
{
   gPluginMgr->AddHandler("TVirtualAuth", "Root", "ROOT::Deprecated::TRootAuth",
      "RootAuth", "ROOT::Deprecated::TRootAuth()");
}
