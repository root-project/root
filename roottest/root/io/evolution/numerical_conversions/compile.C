// The code for compile.C copied from rootalias.C
int compile(const char *what)
{
   TString work, mytype = what, lib, src;
   if (mytype[0] == 'u')
      mytype.Form("unsigned %s", what + 1);
   mytype.ReplaceAll("longlong","long long");

   work.Form(" -DCASE_%s \"-DMYTYPE=%s\" -DWHAT=\\\"%s\\\" ", what, mytype.Data(), what);
   gSystem->AddIncludePath( work );
   lib.Form("lib%s", what);

   return gSystem->CompileMacro("float16.cxx", "k", lib) ? 0 : 1;
}
