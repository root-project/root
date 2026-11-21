import ROOT

ROOT.gInterpreter.Declare(r"""

struct S{};

std::map<std::string, std::vector<std::unique_ptr<S>>> mymap;
mymap["a"] = std::vector<std::unique_ptr<S>>();
""")

for key, value in ROOT.mymap:
    print(key, value)
