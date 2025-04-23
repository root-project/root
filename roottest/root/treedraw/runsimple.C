{
   TFile file("hsimple.root");
   TCanvas c1;
   ntuple->Draw("px:py");
   c1.SaveAs("hsimple.ps");
   Bool_t res = ComparePostscript("hsimple.ps.ref","hsimple.ps");
   if (!res) std::cout << "hsimple.ps and hsimple.ps.ref are different\n";
   return !res;
}
