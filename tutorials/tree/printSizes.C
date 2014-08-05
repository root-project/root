#include "TTree.h"
#include "TBranch.h"
#include "Riostream.h"
#include "TMemFile.h"
#include "TKey.h"
#include "TBranchRef.h"

//
// This macro can be used to get aggregate information on the size
// take on disk or in memory by the various branches in a TTree.
// For example:
/*

root [] printTreeSummary(tree);
The TTree "T" takes 3764343 bytes on disk
  It's branch "event" takes 3760313 bytes on disk

root [] printBranchSummary(tree->GetBranch("event"));
The branch "event" takes 3760313 bytes on disk
  It's sub-branch "TObject" takes 581 bytes on disk
  It's sub-branch "fType[20]" takes 640 bytes on disk
  It's sub-branch "fEventName" takes 855 bytes on disk
  It's sub-branch "fNtrack" takes 506 bytes on disk
  It's sub-branch "fNseg" takes 554 bytes on disk
  It's sub-branch "fNvertex" takes 507 bytes on disk
  It's sub-branch "fFlag" takes 420 bytes on disk
  It's sub-branch "fTemperature" takes 738 bytes on disk
  It's sub-branch "fMeasures[10]" takes 1856 bytes on disk
  It's sub-branch "fMatrix[4][4]" takes 4563 bytes on disk
  It's sub-branch "fClosestDistance" takes 2881 bytes on disk
  It's sub-branch "fEvtHdr" takes 847 bytes on disk
  It's sub-branch "fTracks" takes 3673982 bytes on disk
  It's sub-branch "fHighPt" takes 59640 bytes on disk
  It's sub-branch "fMuons" takes 1656 bytes on disk
  It's sub-branch "fLastTrack" takes 785 bytes on disk
  It's sub-branch "fWebHistogram" takes 596 bytes on disk
  It's sub-branch "fH" takes 10076 bytes on disk
  It's sub-branch "fTriggerBits" takes 1699 bytes on disk
  It's sub-branch "fIsValid" takes 366 bytes on disk

 */

Long64_t GetTotalSize(TBranch * b, bool ondisk, bool inclusive);
Long64_t GetBasketSize(TBranch * b, bool ondisk, bool inclusive);

Long64_t GetBasketSize(TObjArray * branches, bool ondisk, bool inclusive) {
   Long64_t result = 0;
   size_t n = branches->GetEntries();
   for( size_t i = 0; i < n; ++ i ) {
      result += GetBasketSize( dynamic_cast<TBranch*>( branches->At( i ) ), ondisk, inclusive );
   }
   return result;
}

Long64_t GetBasketSize(TBranch * b, bool ondisk, bool inclusive) {
   Long64_t result = 0;
   if (b) {
      if (ondisk && b->GetZipBytes() > 0) {
         result = b->GetZipBytes();
      } else {
         result = b->GetTotBytes();
      }
      if (inclusive) {
         result += GetBasketSize(b->GetListOfBranches(), ondisk, true);
      }
      return result;
   }
   return result;
}

Long64_t GetTotalSize( TBranch * br, bool ondisk, bool inclusive ) {
   TMemFile f("buffer","CREATE");
   if (br->GetTree()->GetCurrentFile()) {
      f.SetCompressionSettings(br->GetTree()->GetCurrentFile()->GetCompressionSettings());
   }
   f.WriteObject(br,"thisbranch");
   TKey* key = f.GetKey("thisbranch");
   Long64_t size;
   if (ondisk)
      size = key->GetNbytes();
   else
      size = key->GetObjlen();
   return GetBasketSize(br, ondisk, inclusive) + size;
}

Long64_t GetTotalSize( TObjArray * branches, bool ondisk ) {
   Long64_t result = 0;
   size_t n = branches->GetEntries();
   for( size_t i = 0; i < n; ++ i ) {
      result += GetTotalSize( dynamic_cast<TBranch*>( branches->At( i ) ), ondisk, true );
      cerr << "After " << branches->At( i )->GetName() << " " << result << endl;
   }
   return result;
}

Long64_t GetTotalSize(TTree *t, bool ondisk) {
   TKey *key = 0;
   if (t->GetDirectory()) {
      key = t->GetDirectory()->GetKey(t->GetName());
   }
   Long64_t ondiskSize = 0;
   Long64_t totalSize = 0;
   if (key) {
      ondiskSize = key->GetNbytes();
      totalSize = key->GetObjlen();
   } else {
      TMemFile f("buffer","CREATE");
      if (t->GetCurrentFile()) {
         f.SetCompressionSettings(t->GetCurrentFile()->GetCompressionSettings());
      }
      f.WriteTObject(t);
      key = f.GetKey(t->GetName());
      ondiskSize = key->GetNbytes();
      totalSize = key->GetObjlen();
   }
   if (t->GetBranchRef() ) {
      if (ondisk) {
         ondiskSize += GetBasketSize(t->GetBranchRef(), true, true);
      } else {
         totalSize += GetBasketSize(t->GetBranchRef(), false, true);
      }
   }
   if (ondisk) {
      return ondiskSize + GetBasketSize(t->GetListOfBranches(), /* ondisk */ true, /* inclusive */ true);
   } else {
      return totalSize + GetBasketSize(t->GetListOfBranches(), /* ondisk */ false, /* inclusive */ true);
   }
}

Long64_t sizeOnDisk(TTree *t) {
   // Return the size on disk on this TTree.

   return GetTotalSize(t, true);
}

Long64_t sizeOnDisk(TBranch *branch, bool inclusive)
{
   // Return the size on disk on this branch.
   // If 'inclusive' is true, include also the size
   // of all its sub-branches.

   return GetTotalSize(branch, true, inclusive);
}

void printBranchSummary(TBranch *br)
{
   cout << "The branch \"" << br->GetName() << "\" takes " << sizeOnDisk(br,true) << " bytes on disk\n";
   size_t n = br->GetListOfBranches()->GetEntries();
   for( size_t i = 0; i < n; ++ i ) {
      TBranch *subbr = dynamic_cast<TBranch*>(br->GetListOfBranches()->At(i));
      cout << "  It's sub-branch \"" << subbr->GetName() << "\" takes " << sizeOnDisk(subbr,true) << " bytes on disk\n";
   }
}

void printTreeSummary(TTree *t)
{
   cout << "The TTree \"" << t->GetName() << "\" takes " << sizeOnDisk(t) << " bytes on disk\n";
   size_t n = t->GetListOfBranches()->GetEntries();
   for( size_t i = 0; i < n; ++ i ) {
      TBranch *br =dynamic_cast<TBranch*>(t->GetListOfBranches()->At(i));
      cout << "  It's branch \"" << br->GetName() << "\" takes " << sizeOnDisk(br,true) << " bytes on disk\n";
   }
}

void printSizes() {}
