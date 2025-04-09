int checkContent(TTree &c)
{

   TTreeReader r(&c);

   TTreeReaderValue<int> v(r, "Tag.fTag");
   r.Next();

   if (*v != 1) {
      std::cerr << "Expect 1 for Tag.fTag but got " << *v << std::endl;
      return 1;
   }
   return 0;
}

int oddName()
{

   TFile f("root-10046.root");
   auto t = f.Get<TTree>("miniTree");

   std::cout << "TTree: " << std::endl;
   auto ret = checkContent(*t);


   TChain c("miniTree");
   c.Add("root-10046.root");

   std::cout << "Chain: " << std::endl;
   ret += checkContent(c);

   return ret;
}
