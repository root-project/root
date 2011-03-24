void add() {
// Fill out the code of the actual test
   cout << "TChain::Add\n";
   cout << "Trying subdir.root/data*.root\n";
   TChain c("tree");
   c.Add("subdir.root/data*.root");
   c.ls("noaddr");
   c.Print();
   cout << "Trying subdir.root/data*.root/tree\n";
   TChain c2("tree");
   c2.Add("subdir.root/data*.root/tree");
   c2.ls("noaddr");
   c2.Print();
   cout << "Trying subdir.root/data1.root/tree\n";
   TChain c3("tree");
   c3.Add("subdir.root/data1.root/tree");
   c3.ls("noaddr");
   c3.Print();
   cout << "Trying subdir.root/data2.root\n";
   TChain c4("tree");
   c4.Add("subdir.root/data2.root");
   c4.ls("noaddr");
   c4.Print();
}

void addfile() {
// Fill out the code of the actual test
   cout << "TChain::AddFile\n";
   cout << "Trying subdir.root/data1.root with a tree name\n";
   TChain c3("tree");
   c3.AddFile("subdir.root/data1.root",TChain::kBigNumber,"tree");
   c3.ls("noaddr");
   c3.Print();
   cout << "Trying subdir.root/data2.root\n";
   TChain c4("tree");
   c4.AddFile("subdir.root/data2.root");
   c4.ls("noaddr");
   c4.Print();
}

void runsubdir() 
{
   addfile();
   add();
}
