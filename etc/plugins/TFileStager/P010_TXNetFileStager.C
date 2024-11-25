void P010_TXNetFileStager()
{
   gPluginMgr->AddHandler("TFileStager", "^[x]?root:", "TNetXNGFileStager",
      "NetxNG", "TNetXNGFileStager(const char *)");
}
