//  example of macro to add two histogram files containing the same histograms
//  (in a directory structure) or ntuples/trees.
//  Histograms are added in memory as well as profile histograms.
//  ntuples and trees are merged.
//  The resulting histograms are saved into a new file.
//  original implementation : Rene Brun
//  extensions by Dirk Geppert to support files with sub-directories

//______________________________________________________________________
// give the list of files below. Last file must be a NULL string
const char *cfiles[] = {
  "file1.root",
  "file2.root",
  ""};
const char *outfile="file.root";
//______________________________________________________________________

TFile   *fnew;
TList   *flist;
TFile   *afile, *file1;

TH1     *h1, *h2;
TTree   *t1, *t2;
TObject *obj;
TKey    *key;

void AddRecursive(TDirectory *root,TDirectory* node);
//______________________________________________________________________
//
//
//
//______________________________________________________________________
void hadd() {

  // create the result file
  fnew = new TFile(outfile,"RECREATE");

  //create a support list for the input files
  flist = new TList();

  //open all input files and insert them in the list of files
  Int_t nfiles = 0;
  while (strlen(cfiles[nfiles])) {
    afile = new TFile(cfiles[nfiles]);
    flist->Add(afile);
    nfiles++;
  }

  //Get a pointer to the first file
  afile = file1 = (TFile*)flist->First();

  AddRecursive(fnew,file1);

  //fnew->ls();
  fnew->Write();
  fnew->Close();
  delete fnew;
  flist->Delete();
  delete flist;
}

//______________________________________________________________________
//
//
//
//______________________________________________________________________
void AddRecursive(TDirectory *root,TDirectory* node) {

  static TDirectory *dact;

  TDirectory *dirsav;

  //We create an iterator to loop on all objects(keys) of first file
  TIter nextkey(node->GetListOfKeys());
  while (key = (TKey*)nextkey()) {
    node->cd();
    obj = key->ReadObj();
    if (obj->IsA()->InheritsFrom("TTree")) { //case of a TTree or TNtuple
      t1 = (TTree*)obj;
      // this part still to be implemented
      // use TChain::Merge instead
    } elseif(obj->IsA()->InheritsFrom("TH1")) { //case of TH1 or TProfile
      h1 = (TH1*)obj;
      afile = (TFile*)flist->After(file1);
      while (afile) { //loop on all files starting at second file
        char* base=strstr(root->GetPath(),":"); base+=2;
        //printf("base=%s\n",base);

        dirsav=gDirectory;
        afile->cd(base);
        h2 = (TH1*)gDirectory->Get(h1->GetName());
        dirsav->cd();
        if (h2) { // here we should check that we can add
          //printf("adding histo %s to %s\n",h2->GetName(),h1->GetName());
          h1->Add(h2);
          delete h2;
        }
        afile = (TFile*)flist->After(afile);
      }
    } elseif(obj->IsA()->InheritsFrom("TDirectory")) { //case of TDirectory
      // recursion
      // printf("Found TDirectory name=%s title=%s\n",
      //     obj->GetName(),obj->GetTitle());
      root->cd();
      dact=root->mkdir(obj->GetName(),obj->GetTitle());
      dact->cd();
      TObject *objsave = obj;
      TKey    *keysave = key;      
      AddRecursive(dact,(TDirectory*)obj);
      obj = objsave;
      key = keysave;
    } else { //another object
      printf("anotherobjname=%s, title=%s\n",obj->GetName(),obj->GetTitle());
    }

    // write node object, modified or not into fnew
    if (obj) {
      root->cd();
      obj->Write(key->GetName());
      delete obj;
      obj=NULL;
    }
  }
  root->cd();
}



