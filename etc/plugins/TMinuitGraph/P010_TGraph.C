void P010_TGraph()
{
   gPluginMgr->AddHandler("TMinuitGraph", "*", "TGraph",
      "Graf", "TGraph(Int_t,const Double_t*,const Double_t*)");
}
