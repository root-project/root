void P010_TXNetFileStager()
{
   gPluginMgr->AddHandler("TFileStager", "^root:", "TXNetFileStager",
      "Netx", "TXNetFileStager(const char *)");
}
