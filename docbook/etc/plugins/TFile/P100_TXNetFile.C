void P100_TXNetFile()
{
   gPluginMgr->AddHandler("TFile", "^root:", "TXNetFile",
      "Netx", "TXNetFile(const char*,Option_t*,const char*,Int_t,Int_t,Bool_t)");
}
