{
TFile *ggss207 = TFile::Open("ggss207.root");
analysis->Scan("Lept_1:Lept_2","Lept_1>=0&&Lept_2!=0");
}
