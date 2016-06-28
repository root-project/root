void P020_TPgSQLServer()
{
   gPluginMgr->AddHandler("TSQLServer", "^pgsql:", "TPgSQLServer",
      "RPgSQL", "TPgSQLServer(const char*,const char*,const char*)");
}
