// The code for compile.C copied from rootalias.C
int compile(const char *what)
{
   TString work, mytype = what, lib;

   work.Form(" -DCASE_%s \"-DMYTYPE=%s\" -DWHAT=\\\"%s\\\" ", what, mytype.Data(), what);
   gSystem->AddIncludePath( work );
   lib.Form("lib%s", what);

   return gSystem->CompileMacro("maptovector.cxx", "k", lib) ? 0 : 1;
}
