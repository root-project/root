void P020_RRawFileNetXNG()
{
   gPluginMgr->AddHandler(
      "ROOT::Internal::RRawFile",
      "^root[s]?:",
      "ROOT::Internal::RRawFileNetXNG",
      "NetxNG",
      "RRawFileNetXNG(std::string_view, ROOT::Internal::RRawFile::ROptions)");
}
