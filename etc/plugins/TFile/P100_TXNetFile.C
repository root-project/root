void P100_TXNetFile()
{
   if (!gEnv->GetValue( "XNet.UseOldClient", 0)) {
      gPluginMgr->AddHandler("TFile", "^[x]?root:", "TNetXNGFile",
         "NetxNG", "TNetXNGFile(const char*,Option_t*,const char*,Int_t,Int_t,Bool_t)");
   } else {
      gPluginMgr->AddHandler("TFile", "^[x]?root:", "TXNetFile",
         "Netx", "TXNetFile(const char*,Option_t*,const char*,Int_t,Int_t,Bool_t)");
   }
}
