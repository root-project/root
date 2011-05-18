void P030_TSapDBServer()
{
   gPluginMgr->AddHandler("TSQLServer", "^sapdb:", "TSapDBServer",
      "SapDB", "TSapDBServer(const char*,const char*,const char*)");
}
