{
gROOT->ProcessLine(".L little_v2.C+");
TClass::GetClass("little")->GetStreamerInfo()->ls();
TFile *file = new TFile("little.root");
TClass::GetClass("little")->GetStreamerInfos()->ls();

cout << "\nReading the 'wrapper' object\n";
file->Get("wrapper");
TClass::GetClass("little")->GetStreamerInfos()->ls();
}
