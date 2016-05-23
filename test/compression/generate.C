int generate() {
  TFile* orgfile = TFile::Open("org.root");
  TList* thelist = orgfile->GetListOfKeys();
  TObjLink* thelink = thelist->FirstLink();
  while (thelink) {
    if (orgfile->Get(thelink->GetObject()->GetName())->InheritsFrom("TTree")) {
      TTree* t;
      orgfile->GetObject(thelink->GetObject()->GetName(),t);
      t->MakeClass("parent");
      return 0;
      break;
    }
    thelink = thelink->Next();
  }
  return 1;
}
