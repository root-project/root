TDSet *make_tdset(const Char_t* basedir, Int_t files_per_node, Float_t proc_fraction)
{
   // This script creates a TDSet object that can be used to process
   // the files generated with the make_event_trees.C script.
   // Conventions for file names made by that script are assumed.
   //
   // basedir:        location of files local to proof slaves
   // files_per_node: number of files on each node
   // proc_fraction:  fraction of files to be processed (gt 0 and le 1)

   if (!gProof) {
      cout << "Must Start PROOF before using make_tdset.C" << endl;
      return 0;
   }

   if (!basedir) {
      cout << "'basedir' must not be empty" << endl;
      return 0;
   }

   if (files_per_node <= 0) {
      cout << "files_per_node must be > 0" << endl;
      return kFALSE;
   }

   if (proc_fraction <= 0  || proc_fraction > 1) {
      cout << "proc_fraction must be greater than 0" << endl;
      cout << "and less than or equal to 1" << endl;
      return 0;
   }

   THashList nodelist;
   nodelist.SetOwner();
   TList* l = gProof->GetSlaveInfo();
   for(Int_t i=0 ; i < l->GetSize() ; i++){
      TSlaveInfo* si = dynamic_cast<TSlaveInfo*>(l->At(i));
      if(!nodelist.FindObject(si->fHostName.Data()))
         nodelist.Add(new TObjString(si->fHostName.Data()));
   }

   TDSet *d = new TDSet("TTree","EventTree");
   for(Int_t i=0; i < nodelist.GetSize() ; i++){
      TObjString* node = dynamic_cast<TObjString*>(nodelist.At(i));
      for(Int_t j=1; j <= files_per_node; j++) {
         Float_t val = (i*files_per_node+j)*proc_fraction;
         if (TMath::Abs(val-TMath::Nint(val)) < (1.-proc_fraction)/2.) continue;

         TString filestr = "root://";
         filestr += node->GetName();
         filestr += "/";
         filestr += basedir;
         filestr += "/event_tree_";
         filestr += node->GetName();
         filestr += "_";
         filestr += j;
         filestr += ".root";
         d->Add(filestr);
      }
   }

   return d;
}
