void P010_TXNetFileStager()
{
   if (!gEnv->GetValue("XNet.UseOldClient", 0)) {
      gPluginMgr->AddHandler("TFileStager", "^[x]?root:", "TNetXNGFileStager",
         "NetxNG", "TNetXNGFileStager(const char *)");
   } else {
      gPluginMgr->AddHandler("TFileStager", "^[x]?root:", "TXNetFileStager",
         "Netx", "TXNetFileStager(const char *)");
   }
}
