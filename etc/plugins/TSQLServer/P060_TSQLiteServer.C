void P060_TSQLiteServer()
{
   gPluginMgr->AddHandler("TSQLServer", "^sqlite:", "TSQLiteServer",
      "RSQLite", "TSQLiteServer(const char*,const char*,const char*)");
}
