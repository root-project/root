void P040_TOracleServer()
{
   gPluginMgr->AddHandler("TSQLServer", "^oracle:", "TOracleServer",
      "Oracle", "TOracleServer(const char*,const char*,const char*)");
}
