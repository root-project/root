void runPyAPITest() {
   int eil = gErrorIgnoreLevel;
   gErrorIgnoreLevel = 3000;
   gSystem->Load( "libPyROOT" );
   gErrorIgnoreLevel = eil;

   TObject* o = new TObject;
   void* a = TPython::ObjectProxy_FromVoidPtr( o, "TObject" );
   printf( "OBC:  should be true:  %d\n", TPython::ObjectProxy_Check( a ) );
   printf( "OBCE: should be false: %d\n", TPython::ObjectProxy_CheckExact( a ) );

   Bool_t match = ( o == TPython::ObjectProxy_AsVoidPtr( a ) );
   printf( "VPC:  should be true:  %d\n", match );
}
