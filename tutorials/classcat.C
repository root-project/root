{
// macro generating the Root class categories inheritance trees

   gROOT->Reset();
   gSystem->Load("libProof");
   gSystem->Load("libTreePlayer");
   gSystem->Load("libTreeViewer");
   gSystem->Load("$PYTHIA6/libPythia6");
   gSystem->Load("libEG");
   gSystem->Load("libEGPythia6");
   gSystem->Load("libPhysics");
   gSystem->Load("libGeom");
   
   TCanvas c1("c1");
   TClassTree ct("ClassTree");

   //base classes
   ct.Draw("*TDirectory:*TCollection:TFree:TIter*:TObjLink:TStorage:*TKey:*TEnv:*TSystem:TSystem*:*TStopwatch:TDatime:TTim*TBenchmark*TRandom:TMath:TObjectTable:TStrin*");
   c1.Print("classcat_base.gif");
   c1.Print("classcat_base.ps");

   //container classes
   ct.Draw("*TDirectory:*TCollection:TFree:TIter*:TObjLink:TStorage:*TKey:TObjectTable");
   c1.Print("classcat_container.gif");
   c1.Print("classcat_container.ps");

   //histogram and minimisation classes
   ct.Draw("*TH1:*TFormula:TMinuit:TAxis:TFitter:TVirtualfitter");
   c1.Print("classcat_hist.gif");
   c1.Print("classcat_hist.ps");
   
   //physics vectors classes
   ct.Draw("TFeldmanCousins:TGenPhaseSpace:TLorentz*:TRotation:TVector2D:TVector3D");
   c1.Print("classcat_physics.gif");
   c1.Print("classcat_physics.ps");

   //tree classes
   ct.Draw("*TTree:*TBranch:*TLeaf:*TKey:*TDirectory:TCut*:TEventList:TChain*:TTreeFormula:*TSelector:TTree*");
   c1.Print("classcat_tree.gif");
   c1.Print("classcat_tree.ps");

   //2-d graphics classes
   ct.Draw("*TBox:TStyle:*TVirtualX:TGaxis:*TText:TPostScript:*TFormula:TColor:*TEllipse:*TLine:*TMarker:*TGraph:TPoly*");
   c1.Print("classcat_graf2d.gif");
   c1.Print("classcat_graf2d.ps");

   //pad/canvas graphics classes
   ct.Draw("*TVirtualPad");
   c1.Print("classcat_gpad.gif");
   c1.Print("classcat_gpad.ps");

   //3-d graphics classes
   ct.Draw("*TShape:*TNode:TGeometry:TPainter3dAlgorithms:TView:TPoly*:TRotMatrix:*TMaterial");
   c1.Print("classcat_graf3d.gif");
   c1.Print("classcat_graf3d.ps");
   
   //Detector geometry classes
   ct.Draw("TGeo*");
   c1.Print("classcat_geometry.gif");
   c1.Print("classcat_geometry.ps");

   //GUI classes
   ct.Draw("*TGFrame");
   c1.Print("classcat_gui.gif");
   c1.Print("classcat_gui.ps");

   //Browser, viewer, inspector classes
   ct.Draw("*TBrowser:TTreeViewer:TClassTree");
   c1.Print("classcat_browse.gif");
   c1.Print("classcat_browse.ps");

   //meta classes
   ct.Draw("*TDictionary:TTogg*:*TCint:TMethodCall:*TInterpreter:TClassTable:TObjectTable:TStream*");
   c1.Print("classcat_meta.gif");
   c1.Print("classcat_meta.ps");

   //net classes
   ct.Draw("TInetAddress:*TSocket:*TBuffer:*TMonitor:*TUrl:TNetFile:TWebFile");
   c1.Print("classcat_net.gif");
   c1.Print("classcat_net.ps");

   //linear algebra and matrix classes
   ct.Draw("TMatr*:TLazymatrix:*TVector");
   c1.Print("classcat_linalgebra.gif");
   c1.Print("classcat_linalgebra.ps");

}
