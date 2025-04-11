{
  gErrorIgnoreLevel = kWarning;
  TChain chain("test");
  chain.AddFile("VectorProxyTest.root");
  chain.MakeProxy("VectorProxyTestSel","VectorProxyTest.C");
  chain.Process("VectorProxyTestSel.h+g");
}
