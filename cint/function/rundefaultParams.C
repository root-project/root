{
  // check interpreted funcs (default params)
  cout << "Interpeted code:" << endl;
  gROOT->ProcessLine(".L Params.h");
  gROOT->ProcessLine(".x testDefaultParams.C");
  gROOT->ProcessLine(".U Params.h");

  // check compiled funcs (default params)
  cout << endl;
  cout << "Compiled code:" << endl;
  gROOT->ProcessLine(".L Params.h+");
  gROOT->ProcessLine(".x testDefaultParams.C");
  return 0;
}
