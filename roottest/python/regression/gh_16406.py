import ROOT

ROOT.gInterpreter.ProcessLine("""
class MyClass {
public:
    class MyObj {
    public:
        MyObj(const char* in) : s(in) { }
        TString s;
    };
    void MyMethod(const MyObj& x="hello", bool opt=false, bool opt2=true) {
      std::cout << x.s << " " << opt << " " << opt2 << std::endl;
    }

};
""")

ROOT.MyClass().MyMethod(opt2=False, x="hi")
