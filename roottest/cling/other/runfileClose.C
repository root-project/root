{
// This test a reference to a closed temporary file is not kept internally in CINT.
// This lead to a core dump when running gROOT->ProcessLine from within a library load.
   gSystem->Load("fileClose_C");
}
