#ifndef ROOT_TREE_TEST_BRANCHCOUNTSTRUCT_H
#define ROOT_TREE_TEST_BRANCHCOUNTSTRUCT_H

#include <Rtypes.h>
#include <TObject.h>

#include <algorithm>

// Split object holding a variable-size array of a basic type (TStreamerBasicPointer).
class BCVertex : public TObject {
public:
   Int_t fN = 0;
   Double_t *fX = nullptr; //[fN]

   BCVertex() = default;
   BCVertex(const BCVertex &other) : TObject(other) { CopyArray(other); }
   BCVertex &operator=(const BCVertex &other)
   {
      if (this != &other) {
         TObject::operator=(other);
         CopyArray(other);
      }
      return *this;
   }
   ~BCVertex() override { delete[] fX; }

   void Set(Int_t n)
   {
      delete[] fX;
      fN = n;
      fX = n > 0 ? new Double_t[n] : nullptr;
      for (Int_t i = 0; i < n; ++i)
         fX[i] = i;
   }

private:
   void CopyArray(const BCVertex &other)
   {
      delete[] fX;
      fN = other.fN;
      fX = fN > 0 ? new Double_t[fN] : nullptr;
      std::copy(other.fX, other.fX + fN, fX);
   }

   ClassDefOverride(BCVertex, 1)
};

class BCHit {
public:
   Double_t fE = 0.;

   virtual ~BCHit() = default;

   ClassDef(BCHit, 1)
};

// Split object holding a variable-size array of objects (TStreamerLoop).
class BCHitList : public TObject {
public:
   Int_t fN = 0;
   BCHit *fH = nullptr; //[fN]

   BCHitList() = default;
   BCHitList(const BCHitList &) = delete;
   BCHitList &operator=(const BCHitList &) = delete;
   ~BCHitList() override { delete[] fH; }

   void Set(Int_t n)
   {
      delete[] fH;
      fN = n;
      fH = n > 0 ? new BCHit[n] : nullptr;
      for (Int_t i = 0; i < n; ++i)
         fH[i].fE = i;
   }

   ClassDefOverride(BCHitList, 1)
};

// Embedded split member: the counter sub-branch is named "fV.fN".
class BCEvent : public TObject {
public:
   BCVertex fV;

   void Set(Int_t n) { fV.Set(n); }

   ClassDefOverride(BCEvent, 1)
};

#endif
