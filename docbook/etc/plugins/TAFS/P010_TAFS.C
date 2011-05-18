void P010_TAFS()
{
   gPluginMgr->AddHandler("TAFS", "*", "TAFS",
      "AFSAuth", "TAFS(const char*,const char*,Int_t)");
}
