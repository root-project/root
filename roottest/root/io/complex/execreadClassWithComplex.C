int compare(const char* filename){
   classWithComplex cref(1,2,3,4);
   classWithComplex ccref(11,22,33,44);
   auto f = TFile::Open(filename);
   auto c = (classWithComplex*)f->Get("classWithComplex1");
   auto cc = (classWithComplex*)f->Get("classWithComplex2");

   if (*c != cref && *cc != ccref){
      cout << "ERROR The objects on file differ from the references!\n"
           << cref.GetF() << ", " << cref.GetD() << " vs onfile " << c->GetF() << ", " << c->GetD() << endl
           << ccref.GetF() << ", " << ccref.GetD() << " vs onfile " << cc->GetF() << ", " << cc->GetD() << endl;
      return 1;
   }
   return 0;
}


int execreadClassWithComplex() {

   int res = 0;
   res+=compare("classWithComplex.root");
//    res+=compare("classWithComplex.xml");
   return res;

}
