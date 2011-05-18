void P010_TTreeViewer()
{
   gPluginMgr->AddHandler("TVirtualTreeViewer", "*", "TTreeViewer",
      "TreeViewer", "TTreeViewer(const TTree*)");
}
