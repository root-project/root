template <class T1, class T2> 
void failUnlessEqual( T1 r, T2 e, const char* c = "") { 
  if( r != e ) {
  cout << "Test failed in " << c << " : got " << r << " expected " << e << endl;
  assert(false);
  }
}
template <class T1> 
void failUnless( T1 r, const char* c = "") { 
  if( ! r ) {
  cout << "Test failed in " << c << " : got " << r << endl;
  assert(false);
  }
}


bool test_ObjectInitialization() {
  Final f;
  f.setPattern(10);   // base class Data from Aggregate 
  f.dval.setPattern(0);
  f.dval1.setPattern(1);
  f.dval2.setPattern(2);
  f.dptr = new Data();
  f.dptr->setPattern(3);
  failUnless(f.checkPattern(10),"Direct base class");
  failUnless(f.dval.checkPattern(0), "Direct aggregation");
  failUnless(f.dval1.checkPattern(1),"Aggregation through a base class 1");
  failUnless(f.dval2.checkPattern(2),"Aggregation through a base class 2");
  failUnless(f.dptr->checkPattern(3),"By pointer");
  return true;
}

bool test_WriteObject() {

  Final f0;
  Final f1;
  f1.setPattern(10); 
  f1.dval.setPattern(0);
  f1.dval1.setPattern(1);
  f1.dval2.setPattern(2);
  f1.dptr = new Data();
  f1.dptr->setPattern(3);
  
  Aggregate a0;
  a0.setPattern(10); 
  a0.setFArrayPattern(); 
  a0.setDArrayPattern(3); 

  Data d0;
  d0.setPattern(0);
  d0.transient1 = 111;
  d0.transient2 = 222;
  d0.transient3 = 222;
  
  Data d1;
  d1.setPattern(1);

  TFile fo("data.root","RECREATE");
  failUnless(fo.WriteObject(&f0,"f0"),"writing f0");
  failUnless(fo.WriteObject(&f1,"f1"),"writing f1");
  failUnless(fo.WriteObjectAny(&d0,"Data","d0"),"writing d0");
  failUnless(fo.WriteObjectAny(&d1,"Data","d1"),"writing d1");
  failUnless(fo.WriteObject(&a0,"a0"),"writing a0");

  fo.Close();
  return true;
}
bool test_ReadObject() {

  TFile fi("data.root");
  Final* f;
  fi.GetObject("f1", f);
  failUnless(f->checkPattern(10),"Indirect base class");
  failUnless(f->dval.checkPattern(0), "Direct aggregation");
  failUnless(f->dval1.checkPattern(1),"Aggregation through a base class 1");
  failUnless(f->dval2.checkPattern(2),"Aggregation through a base class 2");
  failUnless(f->dptr->checkPattern(3),"By pointer");

  Data* d = (Data*)fi.FindObjectAny("d1");
  failUnless(d->checkPattern(1),"Direct");

  Aggregate* a = (Aggregate*)fi.FindObjectAny("a0");
  failUnless(a->checkPattern(10),"Aggregate");
  failUnless(a->checkFArrayPattern(),"Aggregate farray[3]");
  //failUnless(a->checkDArrayPattern(),"Aggregate darray");  //Problem with ROOT

  fi.Close();
  return true;
}
bool test_WriteObjectInTree() {
  Data* d      = new Data();
  Aggregate* g = new Aggregate();
  Final* f     = new Final();

  TFile fo("tree.root","RECREATE");
  TTree* tree = new TTree("tree","Test Tree");
  failUnless( tree->Branch("Data","Data",&d, 32000,99), "Creating Data branch");
  failUnless( tree->Branch("Aggregate","Aggregate",&g, 32000,99), "Creating Aggregate branch");
  failUnless( tree->Branch("Final","Final",&f, 32000,99), "Creating Final branch");
  for (int i = 0; i < 2; i++ ) {
    d->setPattern(i);

    g->setPattern(i); 
    g->dval1.setPattern(i+10);
    g->dval2.setPattern(i+20);
    g->dptr = new Data();
    g->dptr->setPattern(i+30);

    f->setPattern(i+10); 
    f->dval.setPattern(i+20);
    f->dval1.setPattern(i+30);
    f->dval2.setPattern(i+40);
    f->dptr = new Data();
    f->dptr->setPattern(i+50);

    failUnless( tree->Fill(), "Filling tree");
  }
  fo.Write();
  fo.Close();
  return true;
}

bool test_ReadObjectInTree() {
  Data* d      = new Data();
  Aggregate* g = new Aggregate();
  Final* f     = new Final(); 

  TFile fi("tree.root");
  TTree* tree = (TTree*)fi.Get("tree");
  failUnless( tree, "Getting tree");
  tree->GetBranch("Data")->SetAddress(&d);
  tree->GetBranch("Aggregate")->SetAddress(&g);
  tree->GetBranch("Final")->SetAddress(&f);
  Long64_t n = tree->GetEntries();
  failUnlessEqual( 2, n, "Number of entries in Tree");
  for ( int i = 0; i < n; i++ ) {
    failUnless( tree->GetEntry(i), "Getting Entry");
    failUnless( d->checkPattern(i), "Data pattern");
    failUnless( g->checkPattern(i), "Aggregate");
    failUnless( g->dval1.checkPattern(i+10), "Aggregate dval1");
    failUnless( g->dval2.checkPattern(i+20), "Aggregate dvla2");
    failUnless( g->dptr->checkPattern(i+30), "Aggregate dptr");
    failUnless( f->checkPattern(i+10), "In Aggregate base");
    failUnless( f->dval.checkPattern(i+20), "Final dval");
    //---The next two cases is currently a BUG in ROOT
    //failUnless( f->dval1.checkPattern(i+30), "Final dval1");
    //failUnless( f->dval2.checkPattern(i+40), "Final dval2");
    failUnless( f->dptr->checkPattern(i+50), "Final dptr");
  }
  fi.Close();
  return true;
}

bool test_ReadTransientData() {

  TFile fi("data.root");

  Data* d = (Data*)fi.FindObjectAny("d0");
  failUnless(d->checkPattern(0),"Direct");
  failUnlessEqual(d->transient1, 1,"");
  failUnlessEqual(d->transient2, 2,"");
  failUnlessEqual(d->transient3, 3,"");

  fi.Close();
  return true;
}

using namespace ROOT::Cintex;

void test_Persistency()
{
  //gSystem->Load("libReflex");
  //gSystem->Load("test_CintexTestRflx");
  //gSystem->Load("libCintex");

  Cintex::SetDebug(0);
  Cintex::Enable();
  
  //gROOT->GetClass("Data")->GetStreamerInfo()->ls();
  cout << "ObjectInitialization: "      << (test_ObjectInitialization()      ? "OK" : "FAIL") << endl;
  cout << "WriteObject:          "      << (test_WriteObject()       ? "OK" : "FAIL") << endl;
  cout << "ReadObject:           "      << (test_ReadObject()        ? "OK" : "FAIL") << endl;
  cout << "WriteObjectInTree:    "      << (test_WriteObjectInTree() ? "OK" : "FAIL") << endl;
  cout << "ReadObjectInTree:     "      << (test_ReadObjectInTree() ? "OK" : "FAIL") << endl;
  cout << "ReadTransientData:    "      << (test_ReadTransientData() ? "OK" : "FAIL") << endl;
  //gROOT->GetClass("Data")->GetStreamerInfo()->ls();
}
