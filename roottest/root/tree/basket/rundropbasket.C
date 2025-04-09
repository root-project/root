{
TFile *_file0 = TFile::Open("DataTest3_cel.root");
gDirectory->cd("DataSet");
TTree *tree; gDirectory->GetObject("TestA1.cel",tree);
cout.flush(); cerr.flush();
tree->Scan("fX","","",3,3);
tree->DropBaskets();
tree->Scan("fX","","",3,3);
#ifdef ClingWorkAroundBrokenUnnamedReturn
int res ; res = 0;
#else
return 0;
#endif
}
