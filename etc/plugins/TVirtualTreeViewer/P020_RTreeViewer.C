void P020_RTreeViewer()
{
   gPluginMgr->AddHandler("TVirtualTreeViewer", "RTreeViewer", "ROOT::RTreeViewer",
      "TreeViewer", "NewViewer(TTree*)");
}
