void P040_TXNetSystem()
{
   if (!gEnv->GetValue("XNet.UseOldClient", 0)) {
      gPluginMgr->AddHandler("TSystem", "^[x]?root:", "TNetXNGSystem",
         "NetxNG", "TNetXNGSystem(const char *,Bool_t)");
   } else {
      gPluginMgr->AddHandler("TSystem", "^[x]?root:", "TXNetSystem",
         "Netx", "TXNetSystem(const char *,Bool_t)");
   }
}
