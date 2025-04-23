{
  // check interpreted funcs (default params)
  cout << "Interpeted code:" << endl;
  gROOT->ProcessLine(".L Params.h");
  gROOT->ProcessLine(".x testDefaultParams.C(\"interpreted\")");
  gROOT->ProcessLine(".U Params.h");

  // check compiled funcs (default params)
  cout << endl;
  cout << "Compiled code:" << endl;
  gROOT->ProcessLine(".L Params.h+");
  gROOT->ProcessLine(".x testDefaultParams.C(\"compiled\")");
  return 0;
}
