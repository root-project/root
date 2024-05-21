TSystem* ROOT_Plugin_TNetXNGSystem(const char *,Bool_t);
TSystem* ROOT_Plugin_TXNetSystem(const char *,Bool_t);

void P040_TXNetSystem()
{
   if (!gEnv->GetValue("XNet.UseOldClient", 0)) {
      gPluginMgr->AddHandler("TSystem", "^[x]?root:", "TNetXNGSystem",
         "NetxNG", "::ROOT_Plugin_TNetXNGSystem(const char *,Bool_t)");
   } else {
      gPluginMgr->AddHandler("TSystem", "^[x]?root:", "TXNetSystem",
         "Netx", "::ROOT_Plugin_TXNetSystem(const char *,Bool_t)");
   }
}
