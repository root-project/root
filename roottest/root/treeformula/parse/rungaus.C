{
gROOT->ProcessLine(".L gausbug.cxx");
TF1 *test = new TF1("test","gaussianfunction(x)",-10.0, 10.0);
#ifdef ClingWorkAroundCallfuncAndConversion
cout << test->Eval(3.2) << endl;
#else
cout << test->Eval(3) << endl;
#endif
gApplication->Terminate(0);
}
