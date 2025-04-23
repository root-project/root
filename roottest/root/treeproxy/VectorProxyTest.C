//================================================================================
void ProxyTest_Begin(TTree*)
{
}

//================================================================================
void ProxyTest_SlaveBegin(TTree*)
{
}

//================================================================================
Bool_t ProxyTest_Notify() { return kTRUE; }

//================================================================================
Bool_t ProxyTest_Process(Long64_t) 
{ 
  for (size_t ii=0; ii<vecOfVecOfInt->size(); ii++) {
    const std::vector<int>& vecOfInt = vecOfVecOfInt->at(ii);
    for (size_t kk=0; kk<vecOfInt.size(); kk++) {
      printf("ii=%2zu kk=%2zu content=%2d\n", ii, kk, vecOfInt.at(kk));
    }
  }
  
  return kTRUE;
}

//================================================================================
void ProxyTest_Terminate()
{
}

//================================================================================
void ProxyTest_SlaveTerminate()
{
}

//================================================================================
Double_t VectorProxyTest() {
  return 0.0;
}
