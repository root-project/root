void P010_TRFIOSystem()
{
   gPluginMgr->AddHandler("TSystem", "^rfio:", "TRFIOSystem",
      "RFIO", "TRFIOSystem()");
   gPluginMgr->AddHandler("TSystem", "^castor:", "TRFIOSystem",
      "RFIO", "TRFIOSystem()");
}
