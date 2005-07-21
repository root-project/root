{
// Fill out the code of the actual test

gROOT->ProcessLine(".L gausbug.cxx");
TF1 *test = new TF1("test","gaussianfunction(x)",-10.0, 10.0);
cout << test->Eval(3) << endl;
return 0;
}
