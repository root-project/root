void P020_TProofFileStager()
{
   gPluginMgr->AddHandler("TFileStager", "^rootd:", "TNetFileStager",
      "Net", "TNetFileStager(const char *)");
   gPluginMgr->AddHandler("TFileStager", "^rootup:", "TNetFileStager",
      "Net", "TNetFileStager(const char *)");
   gPluginMgr->AddHandler("TFileStager", "^roots:", "TNetFileStager",
      "Net", "TNetFileStager(const char *)");
   gPluginMgr->AddHandler("TFileStager", "^rootk:", "TNetFileStager",
      "Net", "TNetFileStager(const char *)");
   gPluginMgr->AddHandler("TFileStager", "^rootg:", "TNetFileStager",
      "Net", "TNetFileStager(const char *)");
   gPluginMgr->AddHandler("TFileStager", "^rooth:", "TNetFileStager",
      "Net", "TNetFileStager(const char *)");
   gPluginMgr->AddHandler("TFileStager", "^rootug:", "TNetFileStager",
      "Net", "TNetFileStager(const char *)");
}
