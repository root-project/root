{
#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
   if (1) {
#endif
gROOT->ProcessLine(".L gausbug.cxx");
TF1 *test = new TF1("test","gaussianfunction(x)",-10.0, 10.0);
cout << test->Eval(3) << endl;
gApplication->Terminate(0);
#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
   }
#endif
}
