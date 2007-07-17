void P020_TPgSQLServer()
{
   gPluginMgr->AddHandler("TSQLServer", "^pgsql:", "TPgSQLServer",
      "PgSQL", "TPgSQLServer(const char*,const char*,const char*)");
}
