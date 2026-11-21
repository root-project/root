import ROOT

ROOT.gInterpreter.Declare("namespace A{namespace B{class C{};}}")

def theFunc(c: ROOT.A.B.C) -> None:
    pass

ROOT.gInterpreter.Declare("void exec(A::B::C& c, const std::function<void(A::B::C)>& callback) {}")
c = ROOT.A.B.C()
ROOT.exec(c, theFunc)