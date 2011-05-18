void P020_TSQLMonitoringWriter()
{
   gPluginMgr->AddHandler("TVirtualMonitoringWriter", "SQL", "TSQLMonitoringWriter",
      "Net", "TSQLMonitoringWriter(const char *,const char *,const char *,const char *)");
}
