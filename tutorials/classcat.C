{
// macro generating the Root class categories inheritance trees

   gROOT->Reset();
   TCanvas c1("c1");
   TClassTree ct("ClassTree");

   //base classes
   ct.Draw("*TDirectory:*TCollection:TFree:TIter*:TObjLink:TStorage:*TKey:*TEnv:*TSystem:TSystem*:*TStopwatch:TDatime:TTim*TBenchmark*TRandom:TMath:TObjectTable:TStrin*");
   c1.Print("html/gif/classcat_base.gif");
   c1.Print("html/ps/classcat_base.ps");

   //container classes
   ct.Draw("*TDirectory:*TCollection:TFree:TIter*:TObjLink:TStorage:*TKey:TObjectTable");
   c1.Print("html/gif/classcat_container.gif");
   c1.Print("html/ps/classcat_container.ps");

   //histogram and minimisation classes
   ct.Draw("*TH1:*TFormula:TMinuit:TAxis");
   c1.Print("html/gif/classcat_hist.gif");
   c1.Print("html/ps/classcat_hist.ps");

   //tree classes
   ct.Draw("*TTree:*TBranch:*TLeaf:*TKey:*TDirectory:TCut*:TEventList:TChain*");
   c1.Print("html/gif/classcat_tree.gif");
   c1.Print("html/ps/classcat_tree.ps");

   //2-d graphics classes
   ct.Draw("*TBox:TStyle:*TVirtualX:TGaxis:*TText:TPostScript:*TFormula:TColor:*TEllipse:*TLine:*TMarker:*TGraph:TPoly*");
   c1.Print("html/gif/classcat_graf2d.gif");
   c1.Print("html/ps/classcat_graf2d.ps");

   //3-d graphics classes
   ct.Draw("*TShape:*TNode:TGeometry:TLego:TView:TPoly*:TRotMatrix:*TMaterial");
   c1.Print("html/gif/classcat_graf3d.gif");
   c1.Print("html/ps/classcat_graf3d.ps");

   //GUI classes
   ct.Draw("*TGFrame");
   c1.Print("html/gif/classcat_gui.gif");
   c1.Print("html/ps/classcat_gui.ps");

   //Browser, viewer, inspector classes
   ct.Draw("*TBrowser:TTreeViewer:TClassTree");
   c1.Print("html/gif/classcat_browse.gif");
   c1.Print("html/ps/classcat_browse.ps");

   //meta classes
   ct.Draw("*TDictionary:TTogg*:*TCint:TMethodCall:*TInterpreter:TClassTable:TObjectTable");
   c1.Print("html/gif/classcat_meta.gif");
   c1.Print("html/ps/classcat_meta.ps");

   //net classes
   ct.Draw("TInetAddress:*TSocket:*TBuffer:*TMonitor:*TUrl:TNetFile:TWebFile");
   c1.Print("html/gif/classcat_net.gif");
   c1.Print("html/ps/classcat_net.ps");

   //linear algebra and matrix classes
   ct.Draw("TMatr*:TLazymatrix:*TVector");
   c1.Print("html/gif/classcat_linalgebra.gif");
   c1.Print("html/ps/classcat_linalgebra.ps");

}
