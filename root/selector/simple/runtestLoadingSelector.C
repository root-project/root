{
gROOT->ProcessLine(".L testSel.C+");
gROOT->ProcessLine(".L testSelector.C");
bool res = runtest();
if (!res) return !res;
gROOT->ProcessLine(".L testSelector.C+");
bool res = runtest();
return !res;
}
