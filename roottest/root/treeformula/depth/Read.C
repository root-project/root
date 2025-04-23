{
gSystem->Load("libTreePlayer");
TFile * file = new TFile("depth.root");
TTree * tree = T;
tree->Draw("line.line2.line3.fX1");
if (tree->GetSelectedRows()==0) {
   gSystem->Exit(1);
}

}
