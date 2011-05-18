void P030_TCastorFile()
{
   gPluginMgr->AddHandler("TFile", "^castor:", "TCastorFile",
      "RCastor", "TCastorFile(const char*,Option_t*,const char*,Int_t,Int_t)");
}
