{

TFile* mcfile = new TFile("f24100001_0000.sntp.R1.7.root","READ");

TTree* mctree = (TTree*)(mcfile->Get("NtpMC"));

mctree -> MakeProxy("analyzeNtp","print.C","");
mctree -> Process("analyzeNtp.h+","",20);


}
