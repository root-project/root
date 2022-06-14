void P010_TApplicationRemote()
{
   gPluginMgr->AddHandler("TApplication", "^remote", "TApplicationRemote",
      "Net", "TApplicationRemote(const char *,Int_t,const char *)");
}
