// ROOT-7739
{
  double a = 2;
  TVectorD b(2);
  gROOT->ProcessLine("a*b");
  gROOT->ProcessLine("b*a");
  exit(0);
}
