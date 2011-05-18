// $Id: make_tdset.C,v 1.9 2007/07/16 09:57:39 rdm Exp $
//
//

#include "Riostream.h"
#include "TDSet.h"
#include "THashList.h"
#include "TObjString.h"
#include "TMap.h"
#include "TProof.h"


TDSet *make_tdset(const char *basedir, Int_t files_per_node)
{
   // This script creates a TDSet object that can be used to process
   // the files generated with the make_event_trees.C script.
   // Conventions for file names made by that script are assumed.
   //
   // basedir:         location of files local to proof slaves
   // files_per_slave: number of files per node

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
      return 0;
   }

   TList* l = gProof->GetListOfSlaveInfos();
   if (!l) {
      cout << "No list of workers received!" << endl;
      return 0;
   }
   TIter nxw(l);
   TSlaveInfo *si = 0;

   THashList nodelist;
   nodelist.SetOwner(kFALSE);
   while ((si = (TSlaveInfo *) nxw())) {
      if (!nodelist.FindObject(si->GetName())) nodelist.Add(new TPair(new TObjString(si->GetName()), si));
   }

   TDSet *d = new TDSet("TTree","EventTree");
   TIter nxu(&nodelist);
   TPair *p = 0;
   si = 0;
   while ((p = (TPair *) nxu())) {
      si = (TSlaveInfo *) p->Value();
      for (Int_t j = 1; j <= files_per_node ; j++) {
         TString filestr;
         if (gProof->IsLite()) {
            filestr += "file://";
         } else {
            filestr += "root://";
            filestr += si->GetName();
            filestr += "/";
         }
         filestr += basedir;
         filestr += "/event_tree_";
         filestr += si->GetName();
         filestr += "_";
         filestr += j;
         filestr += ".root";
         d->Add(filestr);
      }
   }

   return d;
}
