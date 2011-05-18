void P110_THDFSFile()
{
   gPluginMgr->AddHandler("TFile", "^hdfs:", "THDFSFile",
      "HDFS", "THDFSFile(const char*,Option_t*,const char*,Int_t)");
}
