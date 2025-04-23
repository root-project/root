// Functions
auto f0_ = "int f0()\n"
	      "{\n"
	      "   return 3;\n"
	      "// A comment\n"
	      "}";

auto f1_ = "void f1(int a, double b){\n"
          " // pass!\n"
          "}";

auto f2_ = "int f2(int a, double b){return a*b;}";

std::string getFunctionBodyFromName(const char* fname)
{
   auto v = gInterpreter->CreateTemporary();
   gInterpreter->Evaluate(fname,*v);
   return v->ToString();
}

int checkFunc(const char* name, const char* code)
{
	gInterpreter->Declare(code);
	auto fbody = getFunctionBodyFromName(name) ;
	// skip 2 lines
	fbody.erase(0, fbody.find("\n") + 1);
	fbody.erase(0, fbody.find("\n") + 1);
	if (fbody.find(code) != 0) {
		cerr << "Code retrieved and implementation of function " << name << " are not consistent." << endl
		     << "Code retrieved:" << endl
		     << "+++\n" << fbody << "\n+++\n"
		     << "Implementation:" << endl
		     << "+++\n" << code << "\n+++\n";
		return 1;
	}
	return 0;
}

int getFuncBody()
{
    vector<pair<const char*, const char*>> inputFunctions {
    	{"f0", f0_},
    	{"f1", f1_},
    	{"f2", f2_}
    };

    int retcode = 0;
    for (auto nameCode : inputFunctions) {
    	if ( 0 != checkFunc(nameCode.first, nameCode.second)) retcode++;
    }

	return retcode;
}
