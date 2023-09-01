void P020_TApplicationServer()
{
   gPluginMgr->AddHandler("TApplication", "^server", "TApplicationServer",
      "Net", "TApplicationServer(Int_t *,char **,FILE *,const char*)");
}
