{
   // Try to autoload a non existing class. The autoloading should not try
   // crash the system. A reproducer of ROOT-5759.
   gROOT->ProcessLine( "#include \"does_not_exist.h\"" );
   gROOT->ProcessLine("cool d;");
}
