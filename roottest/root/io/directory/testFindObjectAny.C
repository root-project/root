void doit()
{
   TFile* base = new TFile("f.db","recreate");
   TDirectory* a = base->mkdir("a","First Level Dir");
   a->cd();
   TH1D* ha = new TH1D("ha","ha",10,0,1);
   TDirectory* aa = a->mkdir("aa","Second Level Dira");
   aa->cd();
   TH1D* haa = new TH1D("haa","haa",10,0,1);
   
   a->ls();
   
   printf(" a: created@ %p  found@ %p\n", a,base->FindObjectAny("a"));
   printf("ha: created@ %p  found@ %p\n",ha,base->FindObjectAny("ha"));
   printf("ha: created@ %p  --found@ %p\n",ha,base->FindObjectAny("a/ha"));
#ifdef ClingWorkAroundMissingDynamicScope
   TDirectory *k = (TDirectory*)base->FindObjectAny("a");
#else
   k = (TDirectory*)base->FindObjectAny("a");
#endif
   printf("ha: created@ %p  found@ %p\n",ha,k->FindObjectAny("ha"));
   
   printf("aa: created@ %p  found@ %p\n",aa,base->FindObjectAny("aa"));
   printf("aa: created@ %p  --found@ %p\n",aa,base->FindObjectAny("a/aa"));
   printf("aa: created@ %p  found@ %p\n",aa,k->FindObjectAny("aa"));
   
   printf("haa: created@ %p  found@ %p\n",haa,base->FindObjectAny("haa"));
   printf("haa: created@ %p  --found@ %p\n",haa,base->FindObjectAny("aa/haa"));
   printf("haa: created@ %p  --found@ %p\n",haa,base->FindObjectAny("a/aa/haa"));
   printf("haa: created@ %p  found@ %p\n",haa,k->FindObjectAny("haa"));
   printf("haa: created@ %p  --found@ %p\n",haa,k->FindObjectAny("aa/haa"));
#ifdef ClingWorkAroundMissingDynamicScope
   TDirectory *kk = (TDirectory*)k->FindObjectAny("aa");
#else
   kk = (TDirectory*)k->FindObjectAny("aa");
#endif
   printf("haa: created@ %p  found@ %p\n",haa,kk->FindObjectAny("haa"));
   
   base->Write();
   
}
void doit2()
{
   TFile* base = new TFile("f.db","READ");
   
   TH1D *ha = 0;
   TH1D *haa = 0;
   TDirectory *aa = 0;

#ifdef ClingWorkAroundMissingDynamicScope
   TDirectory *a = (TDirectory*)base->Get("a");
#endif
   a->ls();
   
   printf(" a: created@ %p  found@ %p\n", a,base->FindObjectAny("a"));
   printf("ha: created@ %p  found@ %p\n",ha,base->FindObjectAny("ha"));
   printf("ha: created@ %p  --found@ %p\n",ha,base->FindObjectAny("a/ha"));
#ifdef ClingWorkAroundMissingDynamicScope
   TDirectory *k = (TDirectory*)base->FindObjectAny("a");
#else
   k = (TDirectory*)base->FindObjectAny("a");
#endif
   printf("ha: created@ %p  found@ %p\n",ha,k->FindObjectAny("ha"));
   
   printf("aa: created@ %p  found@ %p\n",aa,base->FindObjectAny("aa"));
   printf("aa: created@ %p  --found@ %p\n",aa,base->FindObjectAny("a/aa"));
   printf("aa: created@ %p  found@ %p\n",aa,k->FindObjectAny("aa"));
   
   printf("haa: created@ %p  found@ %p\n",haa,base->FindObjectAny("haa"));
   printf("haa: created@ %p  --found@ %p\n",haa,base->FindObjectAny("aa/haa"));
   printf("haa: created@ %p  --found@ %p\n",haa,base->FindObjectAny("a/aa/haa"));
   printf("haa: created@ %p  found@ %p\n",haa,k->FindObjectAny("haa"));
   printf("haa: created@ %p  --found@ %p\n",haa,k->FindObjectAny("aa/haa"));
#ifdef ClingWorkAroundMissingDynamicScope
   TDirectory *kk = (TDirectory*)k->FindObjectAny("aa");
#else
   kk = (TDirectory*)k->FindObjectAny("aa");
#endif
   printf("haa: created@ %p  found@ %p\n",haa,kk->FindObjectAny("haa"));
   
}

void testing(TObject *orig, TObject *found) 
{
   if (found == 0) {
      cout << "Could not find ";
      if (orig) cout << orig->GetName();
      else cout << "the requested object";
      cout << "\n";
   } else if (orig && orig != found) {
      cout << "Object " << orig->GetName() << " not correctly found!\n";
   }
}

int testFindObjectAny() 
{ 
   TDirectory* db = gROOT->mkdir("db","db"); 
   TDirectory* a = db->mkdir("a","a"); 
   TDirectory* aa = a->mkdir("aa","aa"); 
   aa->cd(); 
   TH1D* haa_new = new TH1D("haa","haa",10,0,1); 
   TH1D* haa_find = (TH1D*)db->FindObjectAny("haa");
#ifdef ClingWorkAroundMissingDynamicScope
   TH1D* haa = haa_find;
#endif
   if (!haa) {
      cout << "haa missing\n";
   } else if (haa_new != haa_find) {
      cout << "haa not found correctly!\n";
   }
   
   TFile* base = new TFile("fdb.root","recreate");
#ifdef ClingReinstateRedeclarationAllowed
   TDirectory* a = base->mkdir("a","First Level Dir");
#else
   a = base->mkdir("a","First Level Dir");
#endif
   a->cd();
   TH1D* ha = new TH1D("ha","ha",10,0,1);
#ifdef ClingReinstateRedeclarationAllowed
   TDirectory* aa = a->mkdir("aa","Second Level Dira");
#else
   aa = a->mkdir("aa","Second Level Dira");
#endif
   aa->cd();
#ifdef ClingWorkAroundMissingDynamicScope
   haa = new TH1D("haa","haa",10,0,1);
#else
   TH1D* haa = new TH1D("haa","haa",10,0,1);
#endif
   
   testing(   a, base->FindObjectAny("a"));
   testing(  ha, base->FindObjectAny("ha"));
   testing(  ha,    a->FindObjectAny("ha"));
   testing(  aa, base->FindObjectAny("aa"));
   testing(  aa,    a->FindObjectAny("aa"));
   testing( haa, base->FindObjectAny("haa"));
   testing( haa,    a->FindObjectAny("haa"));
   testing( haa,   aa->FindObjectAny("haa"));   
   base->Write();
   
   delete base;
   base = TFile::Open("fdb.root","READ");
   testing(   0, base->FindObjectAny("a"));
   testing(   0, base->FindObjectAny("ha"));
   a = (TDirectory*)base->FindObjectAny("a");
   testing(   0,    a->FindObjectAny("ha"));
   testing(   0, base->FindObjectAny("aa"));
   testing(   0,    a->FindObjectAny("aa"));
   testing(   0, base->FindObjectAny("haa"));
   testing(   0,    a->FindObjectAny("haa"));
   aa = (TDirectory*)base->FindObjectAny("aa");
   testing(   0,   aa->FindObjectAny("haa"));
   
   return 0;
}
