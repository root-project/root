// See https://root-forum.cern.ch/t/increase-template-recursion-depth-for-tdataframe-snapshot/25411/5

using TDFDEF = ROOT::Experimental::TDF::TInterface<ROOT::Detail::TDF::TCustomColumnBase>;

TDFDEF defineRecursive(TDFDEF& d, int n)
{
    std::string name = "a_" + std::to_string(n);
    auto d2 = TDFDEF(d.Define(name, [](){return 1.;}));
    if (n == 1) return d2;
    return defineRecursive(d2, n - 1);
}

int test_templateRecursionLimit()
{
    ROOT::Experimental::TDataFrame tdf(1);
    TDFDEF d(tdf.Define("a_0", [](){return 1.;}));

    auto d_final = defineRecursive(d, 96); //66 limit with 1024 on linux

    d_final.Snapshot("t", "test_templateRecursionLimit.root");

    return 0;

}
