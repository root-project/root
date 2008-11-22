{
// make sure to not load the library
gInterpreter->UnloadLibraryMap("def_C");
TFile *test = TFile::Open("test.root");
TTree *T = (TTree*)test->Get("T");
T->Print();
T->Scan("myvar");
T->Scan("arr.chunk.myvar[0]");
 return 0;
}
