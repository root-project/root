void run()
{
// Avoid loading the library
gInterpreter->UnloadLibraryMap("sel_C");
TFile::Open("Event1.root");
auto tree1 = (TTree*) gFile->Get("T1");
TFile::Open("Event2.root");
auto tree2 = (TTree*) gFile->Get("T2");
TFile::Open("Event3.root");
auto tree3 = (TTree*) gFile->Get("T3");
tree1->Process("sel.C","T1");
tree2->Process("sel.C","T2");
tree3->Process("sel.C","T3");

TSelector *sel = TSelector::GetSelector("sel.C");
tree1->Process(sel,"T1");
tree2->Process(sel,"T2");
tree3->Process(sel,"T3");
}
