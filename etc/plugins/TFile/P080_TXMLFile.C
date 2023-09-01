void P080_TXMLFile()
{
   gPluginMgr->AddHandler("TFile", ".+[.]xml$", "TXMLFile",
      "XMLIO", "TXMLFile(const char*,Option_t*,const char*,Int_t)");
}
