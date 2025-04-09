{
vector<TNamed> v;
gInterpreter->GenerateDictionary("vector<TNamed>");
TTree t("t","with vector");
t.Branch("v",&v,32000,0);
v.emplace_back("a","b");
t.Fill();
t.Scan("v@.at(0).GetName()");
auto res = t.Scan("v@.at(0).fName");
if (res == 1) return 0;
else return 1;
}
