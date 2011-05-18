void P060_THDFSSystem()
{
   gPluginMgr->AddHandler("TSystem", "^hdfs:", "THDFSSystem",
      "HDFS", "THDFSSystem()");
}
