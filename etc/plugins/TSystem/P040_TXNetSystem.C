TSystem* ROOT_Plugin_TNetXNGSystem(const char *,Bool_t);

void P040_TXNetSystem()
{
   gPluginMgr->AddHandler("TSystem", "^[x]?root:", "TNetXNGSystem",
      "NetxNG", "::ROOT_Plugin_TNetXNGSystem(const char *,Bool_t)");
}
