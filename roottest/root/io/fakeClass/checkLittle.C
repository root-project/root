{
TFile *file = new TFile("little.root");
TClass::GetClass("little")->GetStreamerInfo()->ls();
gROOT->ProcessLine(".L little_v2.C+");
TClass::GetClass("little")->GetStreamerInfos()->ls();
TClass::GetClass("little")->GetStreamerInfo()->ls();

cout << "\nReading the 'wrapper' object\n";
file->Get("wrapper");
TClass::GetClass("little")->GetStreamerInfos()->ls();
}
