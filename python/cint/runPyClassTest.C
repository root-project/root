{
   gSystem->Load( "libPyROOT" );
   TPython::LoadMacro( "MyPyClass.py" );

   MyPyClass m;
   printf( "string (aap): %s\n", (char*)m.gime( "aap" ) );
   printf( "double (0.123): %.3f\n", (double)m.gime( 0.123 ) );

   return 0;
}
