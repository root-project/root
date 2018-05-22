// See https://root-forum.cern.ch/t/increase-template-recursion-depth-for-tdataframe-snapshot/25411/5

using namespace ROOT::RDF;
using namespace ROOT::Detail::RDF;
RInterface<RLoopManager> defineRecursive(RInterface<RLoopManager> &d, int n)
{
    std::string name = "a_" + std::to_string(n);
    auto d2 = d.Define(name, [](){return 1.;});
    if (n == 1) return d2;
    return defineRecursive(d2, n - 1);
}

int test_templateRecursionLimit()
{
    ROOT::RDataFrame tdf(1);
    auto d = tdf.Define("a_0", [](){return 1.;});

    auto d_final = defineRecursive(d, 96); //66 limit with 1024 on linux

    d_final.Snapshot("t", "test_templateRecursionLimit.root");

    return 0;

}
