void P100_TXNetFile()
{
   gPluginMgr->AddHandler("TFile", "^[x]?root:", "TNetXNGFile",
      "NetxNG", "TNetXNGFile(const char*,Option_t*,const char*,Int_t,Int_t,Bool_t)");
}
