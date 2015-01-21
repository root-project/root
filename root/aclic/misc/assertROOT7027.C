// Checks that we can unload ACLiC'ed sources (ROOT-7027)
// Due to a failure in unloading, this used to provoke:
/*
In file included from input_line_104:16:
./renamed.cxx:1:6: error: redefinition of 'foo1'
void foo1() { return ; }
     ^
./renamed.cxx:1:6: note: previous definition is here
void foo1() { return ; }
     ^
*/

static const char filename1[] = "workfile.cxx";
static const char filename2[] = "renamed.cxx";
void out(const char* code) {
   std::ofstream ofs(filename1);
   ofs << code;
}

int assertROOT7027() {
   out("void foo1() { return ; }");
   gSystem->CompileMacro(filename1, "fgs");

   out("void foo2() { return ; }");
   gSystem->Rename(filename1, filename2);
   gSystem->CompileMacro(filename2, "fgs");
   return 0;
}
