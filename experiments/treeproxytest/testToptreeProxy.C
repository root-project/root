{

TFile* topfile = new TFile("toptree.root","READ");

 TTree* toptree; topfile->GetObject("TopTree",toptree);

 toptree -> MakeProxy("analyzeTop","printToptree.C","","nohist");
toptree -> Process("analyzeTop.h+","",20);


}
