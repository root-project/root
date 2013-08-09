{
    makeTree();
    printAll();
    makeTree("HardTreeFile2.root", 11);
    TChain myChain ("HardTree");
    myChain.Add("./HardTreeFile.root");
    myChain.Add("./HardTreeFile2.root");
    TFile myFile ("HardChainFile.root", "RECREATE");
    myFile.Add(&myChain);
    myFile.Write();
    printChain();
}
