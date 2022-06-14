class TSQLServer;
TSQLServer* ROOT_Plugin_TPgSQLServer(const char*,const char*,const char*);

void P020_TPgSQLServer()
{
   gPluginMgr->AddHandler("TSQLServer", "^pgsql:", "TPgSQLServer",
      "PgSQL", "::ROOT_Plugin_TPgSQLServer(const char*,const char*,const char*)");
}
