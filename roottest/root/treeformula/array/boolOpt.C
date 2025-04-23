{
TFile *ggss207 = TFile::Open("ggss207.root");
#ifdef ClingWorkAroundMissingDynamicScope
TTree *analysis; ggss207->GetObject("analysis",analysis);
#endif
analysis->Scan("Lept_1:Lept_2","Lept_1>=0&&Lept_2!=0");
#ifndef ClingWorkAroundBrokenUnnamedReturn
return 0;
#else
bool res = 0;
#endif
}
