void build(const char *filename) {
   int result = gSystem->CompileMacro(filename,"kc");
   if (result!=0 && result!=1) gApplication->Terminate(1);
}
