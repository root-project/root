{
TFile *test = TFile::Open("test.root");
TTree *T = (TTree*)test->Get("T");
T->Print();
T->Scan("myvar");
T->Scan("arr.chunk.myvar[0]");
}
