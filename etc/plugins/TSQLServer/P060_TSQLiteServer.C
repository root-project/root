void P060_TSQLiteServer()
{
   gPluginMgr->AddHandler("TSQLServer", "^sqlite:", "TSQLiteServer",
      "SQLite", "TSQLiteServer(const char*,const char*,const char*)");
}
