{
TFile *file = new TFile("Event.new.split9.root");
TTree *tree = (TTree*)file->Get("T");
director = TProxyDirector(tree,-1);
//px = TFloatProxy(&director,"px");
//float calc = px;
meas = TArrayIntProxy(&director,"fMeasures");
//float calc = meas[5];
}
