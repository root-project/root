{
TFile *f = new TFile("ggss207.root");
analysis = (TTree*)gFile->Get("analysis");
int v1 = analysis->Draw("Lept_1[3]","Lept_1[3]>10");
int v2 = analysis->Draw(">>evt1","Lept_1[3]>10");
bool good = true;
if (v1 != v2) {
   cerr << "The number of values returned by both version is different!\n";
   cerr << v1 << " vs " << v2 << endl;
   good = false;
}
eventList = evt1;
if (v1 != eventList->GetN() ) {
   cerr << "The number of values in the event list is incorrect!\n";
   cerr << v1 << " vs " << eventList->GetN() << endl;
   good = false;
}
if (!good) gApplication->Terminate(1);
}
