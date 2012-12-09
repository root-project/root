{
#ifdef ClingWorkAroundMissingImplicitAuto
TChain*
#endif
chain = new TChain("CommonContainersTree"); 
chain->Add("tlorentzvec.root");
chain->Process("tlorentzvecProxy.h+");
}


