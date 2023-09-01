void P020_RTreeViewer()
{
   gPluginMgr->AddHandler("TVirtualTreeViewer", "RTreeViewer", "ROOT::Experimental::RTreeViewer",
      "TreeViewer", "NewViewer(TTree*)");
}
