{
// Avoid loading the library
gInterpreter->UnloadLibraryMap("sel_C");
new TFile("Event1.root");
tree1 = T1;
new TFile("Event2.root");
tree2 = T2;
new TFile("Event3.root");
tree3 = T3;
tree1->Process("sel.C","T1");
tree2->Process("sel.C","T2");
tree3->Process("sel.C","T3");

TSelector *sel = TSelector::GetSelector("sel.C");
tree1->Process(sel,"T1");
tree2->Process(sel,"T2");
tree3->Process(sel,"T3");
return 0;
}
