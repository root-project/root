void P040_TXNetSystem()
{
   gPluginMgr->AddHandler("TSystem", "^root:", "TXNetSystem",
      "Netx", "TXNetSystem(const char *,Bool_t)");
}
