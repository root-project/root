void runPyClassTest() {
// The higher warning ignore level is to suppress warnings about
// classes already being in the class table (on Mac).
   int eil = gErrorIgnoreLevel;
   gErrorIgnoreLevel = 3000;
   TPython::LoadMacro( "MyPyClass.py" );

   MyPyClass m;
   printf( "string (aap): %s\n", (const char*)(*(TPyReturn*)m.gime( "aap" )) );
   printf( "double (0.123): %.3f\n", (double)(*(TPyReturn*)m.gime( 0.123 )) );
}
