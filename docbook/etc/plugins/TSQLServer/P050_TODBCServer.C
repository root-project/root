void P050_TODBCServer()
{
   gPluginMgr->AddHandler("TSQLServer", "^odbc:", "TODBCServer",
      "RODBC", "TODBCServer(const char*,const char*,const char*)");
   gPluginMgr->AddHandler("TSQLServer", "^odbcn:", "TODBCServer",
      "RODBC", "TODBCServer(const char*,const char*,const char*)");
   gPluginMgr->AddHandler("TSQLServer", "^odbcd:", "TODBCServer",
      "RODBC", "TODBCServer(const char*,const char*,const char*)");
}
