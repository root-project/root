void P020_TNetFileStager()
{
   gPluginMgr->AddHandler("TFileStager", "^rootd:", "ROOT::Deprecated::TNetFileStager",
      "Net", "ROOT::Deprecated::TNetFileStager(const char *)");
   gPluginMgr->AddHandler("TFileStager", "^rootup:", "ROOT::Deprecated::TNetFileStager",
      "Net", "ROOT::Deprecated::TNetFileStager(const char *)");
   gPluginMgr->AddHandler("TFileStager", "^roots:", "ROOT::Deprecated::TNetFileStager",
      "Net", "ROOT::Deprecated::TNetFileStager(const char *)");
   gPluginMgr->AddHandler("TFileStager", "^rootk:", "ROOT::Deprecated::TNetFileStager",
      "Net", "ROOT::Deprecated::TNetFileStager(const char *)");
   gPluginMgr->AddHandler("TFileStager", "^rootg:", "ROOT::Deprecated::TNetFileStager",
      "Net", "ROOT::Deprecated::TNetFileStager(const char *)");
   gPluginMgr->AddHandler("TFileStager", "^rooth:", "ROOT::Deprecated::TNetFileStager",
      "Net", "ROOT::Deprecated::TNetFileStager(const char *)");
   gPluginMgr->AddHandler("TFileStager", "^rootug:", "ROOT::Deprecated::TNetFileStager",
      "Net", "ROOT::Deprecated::TNetFileStager(const char *)");
}
