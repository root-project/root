{
TFile *file = new TFile("hsimple.root");
TTree *tree = (TTree*)file->Get("ntuple");
director = TProxyDirector(tree,-1);
px = TFloatProxy(&director,"px");
float calc = px;
// meas = TArrayIntProxy(&director,"fMeasures[10]");
}
