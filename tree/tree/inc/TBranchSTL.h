// @(#)root/tree:$Id$
// author: Lukasz Janyst <ljanyst@cern.ch>

//------------------------------------------------------------------------------
// file:   TBranchSTL.h
//------------------------------------------------------------------------------

#ifndef ROOT_TBranchSTL
#define ROOT_TBranchSTL

#include "TBranch.h"
#include "TIndArray.h"

#include <map>
#include <vector>

class TTree;
class TVirtualCollectionProxy;
class TStreamerInfo;
class TBranchElement;

class TBranchSTL: public TBranch {
   public:
      TBranchSTL();
      TBranchSTL( TTree* tree, const char* name,
                  TVirtualCollectionProxy* collProxy,
                  Int_t buffsize, Int_t splitlevel );
      TBranchSTL( TBranch* parent, const char* name,
                  TVirtualCollectionProxy* collProxy,
                  Int_t buffsize, Int_t splitlevel,
                  TStreamerInfo* info, Int_t id );
      virtual ~TBranchSTL();
              void           Browse( TBrowser *b ) override;
              Bool_t         IsFolder() const override;
              const char    *GetClassName() const override { return fClassName.Data(); }
              Int_t          GetExpectedType(TClass *&clptr,EDataType &type) override;
              Int_t          GetEntry(Long64_t entry = 0, Int_t getall = 0) override;
      virtual TStreamerInfo *GetInfo() const;
              void           Print(Option_t* = "") const override;
              void           SetAddress(void* addr) override;

      ClassDefOverride(TBranchSTL, 1) //Branch handling STL collection of pointers

   private:

      void ReadLeavesImpl( TBuffer& b );
      void FillLeavesImpl( TBuffer& b );
              Int_t          FillImpl(ROOT::Internal::TBranchIMTHelper *) override;

      struct ElementBranchHelper_t
      {
         ElementBranchHelper_t():
            fBranch(nullptr), fPointers(nullptr), fId(0),
            fBaseOffset(0), fPosition(0) {}

         TBranchElement*     fBranch;
         std::vector<void*>* fPointers;
         UChar_t             fId;
         UInt_t              fBaseOffset;
         Int_t               fPosition;
      };

      typedef std::map<TClass*, ElementBranchHelper_t> BranchMap_t;
      BranchMap_t fBranchMap;                           ///<! Branch map
      std::vector<ElementBranchHelper_t> fBranchVector; ///<! Branch vector

      TVirtualCollectionProxy* fCollProxy;    ///<! Collection proxy
      TBranch*                 fParent;       ///<! Parent of this branch
      TClass*                  fIndArrayCl;   ///<! Class of the ind array
      TIndArray                fInd;          ///<! Indices
      TString                  fContName;     ///<  Class name of referenced object
      TString                  fClassName;    ///<  Name of the parent class, if we're the data member
      mutable Int_t            fClassVersion; ///<  Version number of the class
      UInt_t                   fClCheckSum;   ///<  Class checksum
      mutable TStreamerInfo   *fInfo;         ///<! The streamer info
      char*                    fObject;       ///<! Pointer to object at address or the
      Int_t                    fID;           ///<  Element serial number in the streamer info
};

#endif // ROOT_TBranchSTL
