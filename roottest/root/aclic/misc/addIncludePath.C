int addIncludePath() {

    const auto gccVersion = gInterpreter->ProcessLine(".x addIncludePathGCCMajor.C+");
    std::string newIncludePath = "-I/usr/include/c++/";
    newIncludePath += std::to_string(gccVersion);
    gSystem->AddIncludePath(newIncludePath.c_str());
    return gROOT->LoadMacro("addIncludePathTest.C+"); // returns 0 in case of success
}
