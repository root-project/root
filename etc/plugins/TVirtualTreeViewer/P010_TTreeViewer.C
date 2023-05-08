void P010_TTreeViewer()
{
   gPluginMgr->AddHandler("TVirtualTreeViewer", "TTreeViewer", "TTreeViewer",
      "TreeViewer", "TTreeViewer(const TTree*)");
}
