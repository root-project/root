void P010_TMonaLisaWriter()
{
   gPluginMgr->AddHandler("TVirtualMonitoringWriter", "MonaLisa", "TMonaLisaWriter",
      "MonaLisa", "TMonaLisaWriter(const char *,const char *,const char *,const char *,const char *)");
}
