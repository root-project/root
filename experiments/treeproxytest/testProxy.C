{

TFile* mcfile = new TFile("f24100001_0000.sntp.R1.7.root","READ");

 TTree* mctree; mcfile->Get("NtpMC",mctree);

mctree -> MakeProxy("analyzeNtp","print.C","","nohist");
mctree -> Process("analyzeNtp.h+","",20);


}
