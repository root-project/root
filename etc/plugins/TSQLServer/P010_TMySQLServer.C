void P010_TMySQLServer()
{
   gPluginMgr->AddHandler("TSQLServer", "^mysql:", "TMySQLServer",
      "RMySQL", "TMySQLServer(const char*,const char*,const char*)");
}
