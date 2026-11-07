{
gInterpreter->ProcessLine("#include \"dupSymJIT.h\"");
gSystem->Load("libdupSymJIT");
gInterpreter->ProcessLine("dupSymJIT(1)");
gInterpreter->Declare("double dupSymJITd(double d) { return d + 42.; }\nint dupSymJIT(int i) { return i + 42; }");
int res = gInterpreter->ProcessLine("dupSymJITd(1.)");
res == 43 ? 0 : 1;
}
