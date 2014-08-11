// @(#)root/tree:$Id$
// Author Lukasz Janyst <ljanyst@cern.ch>  23/01/2008

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBranchSTL                                                           //
//                                                                      //
// A Branch handling STL collection of pointers (vectors, lists, queues,//
// sets and multisets) while storing them in split mode                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TBranchSTL.h"
#include "TList.h"
#include "TBranchElement.h"
#include "TBasket.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include <string>
#include <utility>

#include "TError.h"

ClassImp(TBranchSTL)

//------------------------------------------------------------------------------
TBranchSTL::TBranchSTL():
   fCollProxy( 0 ),
   fParent( 0 ),
   fIndArrayCl( 0 ),
   fClassVersion( 0 ),
   fClCheckSum( 0 ),
   fInfo( 0 ),
   fObject( 0 ),
   fID( -2 )
{
   // Default constructor.

   fIndArrayCl = TClass::GetClass( "TIndArray" );
   fBranchVector.reserve( 25 );
   fNleaves = 0;
   fReadLeaves = (ReadLeaves_t)&TBranchSTL::ReadLeavesImpl;
   fFillLeaves = (FillLeaves_t)&TBranchSTL::FillLeavesImpl;
}

//------------------------------------------------------------------------------
TBranchSTL::TBranchSTL( TTree *tree, const char *name,
                        TVirtualCollectionProxy *collProxy,
                        Int_t buffsize, Int_t splitlevel )
{
   // Normal constructor, called from TTree.

   fTree         = tree;
   fCollProxy    = collProxy;
   fBasketSize   = buffsize;
   fSplitLevel   = splitlevel;
   fContName     = collProxy->GetCollectionClass()->GetName();
   fClCheckSum   = 0;
   fClassVersion = 1;
   fID           = -2;
   fInfo         = 0;
   fMother = this;
   fParent = 0;
   fDirectory = fTree->GetDirectory();
   fFileName = "";
   SetName( name );
   fIndArrayCl = TClass::GetClass( "TIndArray" );
   fBranchVector.reserve( 25 );
   fNleaves = 0;
   fReadLeaves = (ReadLeaves_t)&TBranchSTL::ReadLeavesImpl;
   fFillLeaves = (FillLeaves_t)&TBranchSTL::FillLeavesImpl;

   //---------------------------------------------------------------------------
   // Allocate and initialize the basket control arrays
   //---------------------------------------------------------------------------
   fBasketBytes = new Int_t[fMaxBaskets];
   fBasketEntry = new Long64_t[fMaxBaskets];
   fBasketSeek = new Long64_t[fMaxBaskets];

   for (Int_t i = 0; i < fMaxBaskets; ++i) {
      fBasketBytes[i] = 0;
      fBasketEntry[i] = 0;
      fBasketSeek[i] = 0;
   }

}

//------------------------------------------------------------------------------
TBranchSTL::TBranchSTL( TBranch* parent, const char* name,
                        TVirtualCollectionProxy* collProxy,
                        Int_t buffsize, Int_t splitlevel,
                        TStreamerInfo* info, Int_t id )
{
   // Normal constructor, called from another branch.

   fTree         = parent->GetTree();
   fCollProxy    = collProxy;
   fBasketSize   = buffsize;
   fSplitLevel   = splitlevel;
   fContName     = collProxy->GetCollectionClass()->GetName();
   fClassName    = info->GetClass()->GetName();
   fClassVersion = info->GetClassVersion();
   fClCheckSum   = info->GetClass()->GetCheckSum();
   fInfo         = info;
   fID           = id;
   fMother = parent->GetMother();
   fParent = parent;
   fDirectory = fTree->GetDirectory();
   fFileName = "";
   fNleaves = 0;
   fReadLeaves = (ReadLeaves_t)&TBranchSTL::ReadLeavesImpl;
   fFillLeaves = (ReadLeaves_t)&TBranchSTL::FillLeavesImpl;

   SetName( name );
   fIndArrayCl = TClass::GetClass( "TIndArray" );
   fBranchVector.reserve( 25 );

   //---------------------------------------------------------------------------
   // Allocate and initialize the basket control arrays
   //---------------------------------------------------------------------------
   fBasketBytes = new Int_t[fMaxBaskets];
   fBasketEntry = new Long64_t[fMaxBaskets];
   fBasketSeek = new Long64_t[fMaxBaskets];

   for (Int_t i = 0; i < fMaxBaskets; ++i) {
      fBasketBytes[i] = 0;
      fBasketEntry[i] = 0;
      fBasketSeek[i] = 0;
   }

}

//------------------------------------------------------------------------------
TBranchSTL::~TBranchSTL()
{
   //destructor
   BranchMap_t::iterator brIter;
   for( brIter = fBranchMap.begin(); brIter != fBranchMap.end(); ++brIter ) {
      (*brIter).second.fPointers->clear();
      delete (*brIter).second.fPointers;
   }
}

//------------------------------------------------------------------------------
void TBranchSTL::Browse( TBrowser *b )
{
   //browse a STL branch
   Int_t nbranches = fBranches.GetEntriesFast();
   if (nbranches > 0) {
      TList persistentBranches;
      TBranch* branch=0;
      TIter iB(&fBranches);
      while( (branch = (TBranch*)iB()) )
         persistentBranches.Add(branch);
      persistentBranches.Browse( b );
   }
}

//------------------------------------------------------------------------------
Int_t TBranchSTL::Fill()
{
   //---------------------------------------------------------------------------
   // Cleanup after revious fill
   //---------------------------------------------------------------------------
   BranchMap_t::iterator brIter;
   for( brIter = fBranchMap.begin(); brIter != fBranchMap.end(); ++brIter )
      (*brIter).second.fPointers->clear();

   //---------------------------------------------------------------------------
   // Check if we're dealing with the null pointer here
   //---------------------------------------------------------------------------
   if( fAddress != fObject ) {
      //------------------------------------------------------------------------
      // We have received a zero pointer - treat it as an empty collection
      //------------------------------------------------------------------------
      if( fObject == 0 ) {
         Int_t bytes      = 0;
         Int_t totalBytes = 0;

         //---------------------------------------------------------------------
         // Store the indices
         //---------------------------------------------------------------------
         fInd.SetNumItems( 0 );
         bytes = TBranch::Fill();

         if( bytes < 0 ) {
            Error( "Fill", "The IO error while writing the indices!");
            return -1;
         }
         totalBytes += bytes;

         //---------------------------------------------------------------------
         // Store the branches
         //---------------------------------------------------------------------
         for( Int_t i = 0; i < fBranches.GetEntriesFast(); ++i ) {
            TBranch *br = (TBranch *)fBranches.UncheckedAt(i);
            bytes = br->Fill();
            if( bytes < 0 ) {
               Error( "Fill", "The IO error while writing the branch %s!", br->GetName() );
               return -1;
            }
            totalBytes += bytes;
         }
         return totalBytes;
      }
   }

   //---------------------------------------------------------------------------
   // Set upt the collection proxy
   //---------------------------------------------------------------------------
   TVirtualCollectionProxy::TPushPop helper( fCollProxy, fObject );
   UInt_t size = fCollProxy->Size();

   //---------------------------------------------------------------------------
   // Set up the container of indices
   //---------------------------------------------------------------------------
   if( fInd.GetCapacity() < size )
      fInd.ClearAndResize( size );

   fInd.SetNumItems( size );

   //---------------------------------------------------------------------------
   // Loop over the elements and create branches as needed
   //---------------------------------------------------------------------------
   TClass*               cl         = fCollProxy->GetValueClass();
   TClass*               actClass   = 0;
   TClass*               vectClass  = 0;
   char*                 element    = 0;
   std::vector<void*>*   elPointers = 0;
   TBranchElement*       elBranch   = 0;
   UInt_t                elOffset   = 0;
   UChar_t               maxID      = fBranches.GetEntriesFast()+1;
   UChar_t               elID;
   ElementBranchHelper_t bHelper;
   Int_t                 totalBytes = 0;
   Int_t                 bytes      = 0;
   TString               brName;

   for( UInt_t i = 0; i < size; ++i ) {
      //------------------------------------------------------------------------
      // Determine the actual class of current element
      //------------------------------------------------------------------------
      element = *(char**)fCollProxy->At( i );
      if( !element ) {
         fInd.At(i) = 0;
         continue;
      }

      // coverity[dereference] since this is a TBranchSTL by definition the collection contains pointers to objects.
      actClass = cl->GetActualClass( element );
      brIter = fBranchMap.find( actClass );

      //------------------------------------------------------------------------
      // The branch was not found - create a new one
      //------------------------------------------------------------------------
      if( brIter == fBranchMap.end() ) {
         //---------------------------------------------------------------------
         // Find the dictionary for vector of pointers to this class
         //---------------------------------------------------------------------
         std::string vectClName("vector<");
         vectClName += actClass->GetName() + std::string("*>");
         vectClass = TClass::GetClass( vectClName.c_str() );
         if( !vectClass ) {
            Warning( "Fill", "Unable to find dictionary for class %s", vectClName.c_str() );
            continue;
         }

         //---------------------------------------------------------------------
         // Create the vector of pointers to objects of this type and new branch
         // for it
         //---------------------------------------------------------------------
         elPointers = new std::vector<void*>();//vectClass->GetCollectionProxy()->New();
         if (fName.Length() && fName[fName.Length()-1]=='.') {
            // The top level branch already has a trailing dot.
            brName.Form( "%s%s", GetName(), actClass->GetName() );
         } else {
            brName.Form( "%s.%s", GetName(), actClass->GetName() );
         }
         elBranch   = new TBranchElement( this, brName,
                                          vectClass->GetCollectionProxy(),
                                          fBasketSize, fSplitLevel-1  );
         elID = maxID++;
         elBranch->SetFirstEntry( fEntryNumber );


         fBranches.Add( elBranch );

         bHelper.fId         = elID;
         bHelper.fBranch     = elBranch;
         bHelper.fPointers   = elPointers;
         bHelper.fBaseOffset = actClass->GetBaseClassOffset( cl );

         // This copies the value of fPointes into the vector and transfers
         // its ownership to the vector.  It will be deleted in ~TBranch.
         brIter = fBranchMap.insert(std::make_pair(actClass, bHelper ) ).first;
         elBranch->SetAddress( &((*brIter).second.fPointers) );
      }
      //------------------------------------------------------------------------
      // The branch for this type already exists - set up the pointers
      //------------------------------------------------------------------------
      else {
         elPointers = (*brIter).second.fPointers;
         elBranch   = (*brIter).second.fBranch;
         elID       = (*brIter).second.fId;
         elOffset   = (*brIter).second.fBaseOffset;
      }

      //-------------------------------------------------------------------------
      // Add the element to the appropriate vector
      //-------------------------------------------------------------------------
      elPointers->push_back( element + elOffset );
      fInd.At(i) = elID;
   }

   //----------------------------------------------------------------------------
   // Store the indices
   //----------------------------------------------------------------------------
   bytes = TBranch::Fill();
   if( bytes < 0 ) {
      Error( "Fill", "The IO error while writing the indices!");
      return -1;
   }
   totalBytes += bytes;

   //----------------------------------------------------------------------------
   // Fill the branches
   //----------------------------------------------------------------------------
   for( Int_t i = 0; i < fBranches.GetEntriesFast(); ++i ) {
      TBranch *br = (TBranch *)fBranches.UncheckedAt(i);
      bytes = br->Fill();
      if( bytes < 0 ) {
         Error( "Fill", "The IO error while writing the branch %s!", br->GetName() );
         return -1;
      }
      totalBytes += bytes;
   }

   return totalBytes;
}

//------------------------------------------------------------------------------
Int_t TBranchSTL::GetEntry( Long64_t entry, Int_t getall )
{
   //---------------------------------------------------------------------------
   // Check if we should be doing this at all
   //---------------------------------------------------------------------------
   if( TestBit( kDoNotProcess ) && !getall )
      return 0;

   if ( (entry < fFirstEntry) || (entry >= fEntryNumber) )
      return 0;

   if( !fAddress )
      return 0;

   //---------------------------------------------------------------------------
   // Set up the collection proxy
   //---------------------------------------------------------------------------
   if( !fCollProxy ) {
      TClass *cl = TClass::GetClass( fContName );

      if( !cl ) {
         Error( "GetEntry", "Dictionary class not found for: %s", fContName.Data() );
         return -1;
      }

      fCollProxy = cl->GetCollectionProxy();
      if( !fCollProxy ) {
         Error( "GetEntry", "No collection proxy!"  );
         return -1;
      }
   }

   //---------------------------------------------------------------------------
   // Get the indices
   //---------------------------------------------------------------------------
   Int_t totalBytes = 0;
   Int_t bytes = TBranch::GetEntry( entry, getall );
   totalBytes += bytes;

   if( bytes == 0 )
      return 0;

   if( bytes < 0 ) {
      Error( "GetEntry", "IO error! Unable to get the indices!"  );
      return -1;
   }

   Int_t size = fInd.GetNumItems();

   //---------------------------------------------------------------------------
   // Set up vector pointers
   //---------------------------------------------------------------------------
   UInt_t nBranches = fBranches.GetEntriesFast();
   TClass*               elClass   = fCollProxy->GetValueClass();
   TClass*               tmpClass  = 0;

   if( fBranchVector.size() < nBranches )
      fBranchVector.resize( nBranches );

   //---------------------------------------------------------------------------
   // Create the object
   //---------------------------------------------------------------------------
   if( fAddress != fObject ) {
      *((void **)fAddress) = fCollProxy->New();
      fObject = *(char**)fAddress;
   }
   TVirtualCollectionProxy::TPushPop helper( fCollProxy, fObject );
   void* env = fCollProxy->Allocate( size, kTRUE );

   //---------------------------------------------------------------------------
   // Process entries
   //---------------------------------------------------------------------------
   UChar_t             index      = 0;
   void**              element    = 0;
   std::vector<void*>* elemVect   = 0;
   TBranchElement*     elemBranch = 0;

   for( Int_t i = 0; i < size; ++i ) {
      element = (void**)fCollProxy->At(i);
      index   = fInd.At(i);

      //------------------------------------------------------------------------
      // The case of zero pointers
      //------------------------------------------------------------------------
      if( index == 0 ) {
         *element = 0;
         continue;
      }

      //-------------------------------------------------------------------------
      // Index out of range!
      //------------------------------------------------------------------------
      if( index > nBranches ) {
         Error( "GetEntry", "Index %d out of range, unable to find the branch, setting pointer to 0",
                index );
         *element = 0;
         continue;
      }

      //------------------------------------------------------------------------
      // Load unloaded branch
      //------------------------------------------------------------------------
      index--;
      elemVect = fBranchVector[index].fPointers;
      if( !elemVect ) {
         elemBranch = (TBranchElement *)fBranches.UncheckedAt(index);
         elemBranch->SetAddress( &(fBranchVector[index].fPointers) );

         bytes = elemBranch->GetEntry( entry, getall );

         if( bytes == 0 ) {
            Error( "GetEntry", "No entry for index %d, setting pointer to 0", index );
            *element = 0;
            fBranchVector[index].fPosition++;
            continue;
         }

         if( bytes <= 0 ) {
            Error( "GetEntry", "I/O error while getting entry for index %d, setting pointer to 0", index );
            *element = 0;
            fBranchVector[index].fPosition++;
            continue;
         }
         totalBytes += bytes;
         elemVect = fBranchVector[index].fPointers;

         //---------------------------------------------------------------------
         // Calculate the base class offset
         //---------------------------------------------------------------------
         TVirtualCollectionProxy *proxy = elemBranch->GetCollectionProxy();
         if (!proxy) {
            proxy =  TClass::GetClass(elemBranch->GetClassName())->GetCollectionProxy();
         }
         if (proxy) {
            tmpClass = proxy->GetValueClass();
            if (tmpClass && elClass) {
               fBranchVector[index].fBaseOffset = tmpClass->GetBaseClassOffset( elClass );
               fBranchVector[index].fPosition = 0;
            } else {
               Error("GetEntry","Missing TClass for %s (%s)",elemBranch->GetName(),elemBranch->GetClassName());
            }
         } else {
            Error("GetEntry","Missing CollectionProxy for %s (%s)",elemBranch->GetName(),elemBranch->GetClassName());
         }
      }

      //------------------------------------------------------------------------
      // Set up the element
      //------------------------------------------------------------------------
      *element =  ((char*)(*elemVect)[fBranchVector[index].fPosition++])
        - fBranchVector[index].fBaseOffset;

   }

   fCollProxy->Commit(env);

   //---------------------------------------------------------------------------
   // Cleanup
   //---------------------------------------------------------------------------
   for( UInt_t i = 0; i < fBranchVector.size(); ++i ) {
      delete fBranchVector[i].fPointers;
      fBranchVector[i].fPointers = 0;
   }

   return totalBytes;
}


//______________________________________________________________________________
Int_t TBranchSTL::GetExpectedType(TClass *&expectedClass,EDataType &expectedType)
{
   // Fill expectedClass and expectedType with information on the data type of the
   // object/values contained in this branch (and thus the type of pointers
   // expected to be passed to Set[Branch]Address
   // return 0 in case of success and > 0 in case of failure.

   expectedClass = 0;
   expectedType = kOther_t;

   if (fID < 0) {
      expectedClass = TClass::GetClass( fContName );
   } else {
      // Case of an object data member.  Here we allow for the
      // variable name to be ommitted.  Eg, for Event.root with split
      // level 1 or above  Draw("GetXaxis") is the same as Draw("fH.GetXaxis()")
      TStreamerElement* element = GetInfo()->GetElement(fID);
      if (element) {
         expectedClass = element->GetClassPointer();
         if (!expectedClass) {
            Error("GetExpectedType", "TBranchSTL did not find the TClass for %s", element->GetTypeNameBasic());
            return 1;
         }
      } else {
         Error("GetExpectedType", "Did not find the type for %s",GetName());
         return 2;
      }
   }
   return 0;
}

//------------------------------------------------------------------------------
TStreamerInfo* TBranchSTL::GetInfo() const
{
   //---------------------------------------------------------------------------
   // Check if we don't have the streamer info
   //---------------------------------------------------------------------------
   if( !fInfo ) {
      //------------------------------------------------------------------------
      // Get the class info
      //------------------------------------------------------------------------
      TClass *cl = TClass::GetClass( fClassName );

      //------------------------------------------------------------------------
      // Get unoptimized streamer info
      //------------------------------------------------------------------------
      fInfo = (TStreamerInfo*)cl->GetStreamerInfo( fClassVersion );

      //------------------------------------------------------------------------
      // If the checksum is there and we're dealing with the foreign class
      //------------------------------------------------------------------------
      if( fClCheckSum && !cl->IsVersioned() ) {
         //---------------------------------------------------------------------
         // Loop over the infos
         //---------------------------------------------------------------------
         Int_t ninfos = cl->GetStreamerInfos()->GetEntriesFast() - 1;
         for( Int_t i = -1; i < ninfos; ++i ) {
            TVirtualStreamerInfo* info = (TVirtualStreamerInfo*) cl->GetStreamerInfos()->UncheckedAt(i);
            if( !info )
               continue;

            //------------------------------------------------------------------
            // If the checksum matches then retriev the info
            //------------------------------------------------------------------
            if( info->GetCheckSum() == fClCheckSum ) {
               fClassVersion = i;
               fInfo = (TStreamerInfo*)cl->GetStreamerInfo( fClassVersion );
            }
         }
      }
   }
   return fInfo;
}

//------------------------------------------------------------------------------
Bool_t TBranchSTL::IsFolder() const
{
   //branch declared folder if at least one entry

   if( fBranches.GetEntriesFast() >= 1 )
      return kTRUE;
   return kFALSE;
}

//------------------------------------------------------------------------------
void TBranchSTL::Print(const char *option) const
{
   // Print the branch parameters.

   if (strncmp(option,"debugAddress",strlen("debugAddress"))==0) {
      if (strlen(GetName())>24) Printf("%-24s\n%-24s ", GetName(),"");
      else Printf("%-24s ", GetName());

      TBranchElement *parent = dynamic_cast<TBranchElement*>(GetMother()->GetSubBranch(this));
      Int_t ind = parent ? parent->GetListOfBranches()->IndexOf(this) : -1;
      TVirtualStreamerInfo *info = GetInfo();
      Int_t *branchOffset = parent ? parent->GetBranchOffset() : 0;

      Printf("%-16s %2d SplitCollPtr %-16s %-16s %8x %-16s n/a\n",
             info ? info->GetName() : "StreamerInfo unvailable", fID,
             GetClassName(), fParent ? fParent->GetName() : "n/a",
             (branchOffset && parent && ind>=0) ? branchOffset[ind] : 0,
             fObject);
      for( Int_t i = 0; i < fBranches.GetEntriesFast(); ++i ) {
         TBranch *br = (TBranch *)fBranches.UncheckedAt(i);
         br->Print("debugAddressSub");
      }
   } else if (strncmp(option,"debugInfo",strlen("debugInfo"))==0)  {
      Printf("Branch %s uses:\n",GetName());
      if (fID>=0) {
         GetInfo()->GetElement(fID)->ls();
      }
      for (Int_t i = 0; i < fBranches.GetEntriesFast(); ++i) {
         TBranchElement* subbranch = (TBranchElement*)fBranches.At(i);
         subbranch->Print("debugInfoSub");
      }
      return;
   } else {
      TBranch::Print(option);
      for( Int_t i = 0; i < fBranches.GetEntriesFast(); ++i ) {
         TBranch *br = (TBranch *)fBranches.UncheckedAt(i);
         br->Print(option);
      }
   }
}

//------------------------------------------------------------------------------
void TBranchSTL::ReadLeavesImpl( TBuffer& b )
{
   //TO BE DOCUMENTED

   b.ReadClassBuffer( fIndArrayCl, &fInd );
}

//------------------------------------------------------------------------------
void TBranchSTL::FillLeavesImpl( TBuffer& b )
{
   //TO BE DOCUMENTED

   b.WriteClassBuffer( fIndArrayCl, &fInd );
}

//------------------------------------------------------------------------------
void TBranchSTL::SetAddress( void* addr )
{
   //---------------------------------------------------------------------------
   // We are the top level branch
   //---------------------------------------------------------------------------
   if( fID < 0 ) {
      fAddress = (char*)addr;
      fObject  = *(char**)addr;
   }
   //---------------------------------------------------------------------------
   // We are a data member of some other class
   //---------------------------------------------------------------------------
   else {
      //------------------------------------------------------------------------
      // Get the appropriate streamer element
      //------------------------------------------------------------------------
      GetInfo();
      TStreamerElement *el = (TStreamerElement*)fInfo->GetElements()->At( fID );

      //------------------------------------------------------------------------
      // Set up the addresses
      //------------------------------------------------------------------------
      if( el->IsaPointer() ) {
         fAddress = (char*)addr+el->GetOffset();
         fObject  = *(char**)fAddress;
      } else {
         fAddress = (char*)addr+el->GetOffset();
         fObject  = (char*)addr+el->GetOffset();
      }
   }
}
