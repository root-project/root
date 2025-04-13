void build(const char *filename) {
   int result = gSystem->CompileMacro(filename,"kc");
   if (!result) gApplication->Terminate(1);
}
