void ROOT_8197() {
  TFile f("myfile1.root", "recreate");
  TNamed n1("n1", "n1");
  TNamed n2("n2", "n2");
  TNamed n3("n3", "n3");
  TList l;
  l.Add(&n1); l.Add(&n2); l.Add(&n3);
  l.Write("mylist", 1);
 
  std::cout << "** first file **" << std::endl;
  f.ls();
  f.Close();

  gSystem->Exec("rootcp --recreate myfile1.root:mylist myfile2.root:mylist");
  
  TFile f2("myfile2.root");
  std::cout << "** second file **" << std::endl;
  f2.ls();  
}
