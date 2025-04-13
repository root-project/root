{
auto chain = new TChain("CommonContainersTree"); chain->Add("tlorentzvec.root");
chain->MakeProxy("tlorentzvecProxy","analyze.C","cutting.C");
}


