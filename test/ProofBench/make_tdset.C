// $Id: make_tdset.C,v 1.6 2005/02/12 02:14:54 rdm Exp $
//
//

#include "Riostream.h"
#include "TArrayI.h"
#include "TDSet.h"
#include "THashList.h"
#include "TList.h"
#include "TObjString.h"
#include "TProof.h"


TDSet *make_tdset(const Char_t* basedir, Int_t files_per_slave, Int_t max_per_node = 0)
{
   // This script creates a TDSet object that can be used to process
   // the files generated with the make_event_trees.C script.
   // Conventions for file names made by that script are assumed.
   //
   // basedir:         location of files local to proof slaves
   // files_per_slave: number of files per slave to process
   // max_per_node:    maximum available number of file per node

   if (!gProof) {
      cout << "Must Start PROOF before using make_tdset.C" << endl;
      return 0;
   }

   if (!basedir) {
      cout << "'basedir' must not be empty" << endl;
      return 0;
   }

   if (files_per_slave <= 0) {
      cout << "files_per_slave must be > 0" << endl;
      return 0;
   }

   THashList nodelist;
   TArrayI nslaves;
   nodelist.SetOwner();
   TList msdlist;
   msdlist.SetOwner();
   TList* l = gProof->GetSlaveInfo();
   for(Int_t i=0 ; i < l->GetSize() ; i++){
      TSlaveInfo* si = dynamic_cast<TSlaveInfo*>(l->At(i));
      if (si->fStatus != TSlaveInfo::kActive) continue;
      TObjString* host = dynamic_cast<TObjString*>(
                           nodelist.FindObject(si->fHostName.Data()));
      if (host != 0) {
         Int_t index = nodelist.IndexOf(host);
         nslaves[index]++;
      } else {
         nodelist.Add(new TObjString(si->fHostName.Data()));
         msdlist.Add(new TObjString(si->fMsd.Data()));
         nslaves.Set(1+nslaves.GetSize());
         nslaves[nslaves.GetSize()-1] = 1;
      }
   }

   TDSet *d = new TDSet("TTree","EventTree");
   for(Int_t i=0; i < nodelist.GetSize() ; i++){
      TObjString* node = dynamic_cast<TObjString*>(nodelist.At(i));
      TObjString* msd = dynamic_cast<TObjString*>(msdlist.At(i));
      for(Int_t j=1;
          (j <= files_per_slave*nslaves[i])
          && (max_per_node==0 || j<=max_per_node) ;
          j++) {

         TString filestr = "root://";
         filestr += node->GetName();
         filestr += "/";
         filestr += basedir;
         filestr += "/event_tree_";
         filestr += (new TUrl(node->GetName()))->GetHostFQDN();
         filestr += "_";
         filestr += j;
         filestr += ".root";
         if (msd->String().IsNull())
            d->Add(filestr);
         else
            d->Add(filestr,0,0,0,-1,msd->GetName());
      }
   }

   return d;
}
