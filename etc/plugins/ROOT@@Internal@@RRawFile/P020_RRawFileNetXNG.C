void P020_RRawFileNetXNG()
{
   gPluginMgr->AddHandler(
      "ROOT::Internal::RRawFile",
      "^[x]?root[s]?:",
      "ROOT::Internal::RRawFileNetXNG",
      "NetxNG",
      "RRawFileNetXNG(std::string_view, ROOT::Internal::RRawFile::ROptions)");
}
