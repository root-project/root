// @(#)root/geom:$Id$
// Author: Andrei Gheata   30/10/01

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoPatternFinder
#define ROOT_TGeoPatternFinder

#include "TObject.h"

#include <algorithm>
#include <atomic>
#include <vector>

#include "TGeoVolume.h"

class TGeoMatrix;

/// base finder class for patterns. A pattern is specifying a division type
class TGeoPatternFinder : public TObject {
   static std::atomic<UInt_t> fgInstanceCount; //! source of monotonic per-object indices
   UInt_t fIndex{fgInstanceCount++};           //! non-reused index of this finder into the per-thread vector
   mutable std::atomic<Int_t> fGeneration{0};  //! bumped whenever the per-thread state must be rebuilt

public:
   struct ThreadData_t {
      // fMatrix is produced by the virtual CreateMatrix() and registered with (owned by) the
      // geometry manager, so this struct does not own it. All members are trivially movable,
      // which lets the slot live in a resizable thread_local vector.
      TGeoMatrix *fMatrix{nullptr}; //! generic matrix (owned by TGeoManager)
      Int_t fCurrent{-1};           //! current division element
      Int_t fNextIndex{-1};         //! index of next node
      Int_t fInitGen{-1};           //! generation this slot was last initialized for
   };

   /// Per-thread scratch state, owned by the calling thread and indexed by this finder.
   /// Hot path: a TLS read plus an indexed load; the cold rebuild lives in InitThreadSlot().
   /// The vector retains its high-water size until the owning thread exits.
   ThreadData_t &GetThreadData() const
   {
      thread_local std::vector<ThreadData_t> tdata;
      if (tdata.size() <= fIndex)
         tdata.resize(std::max<size_t>(fgInstanceCount.load(std::memory_order_relaxed), fIndex + 1));
      ThreadData_t &td = tdata[fIndex];
      if (td.fInitGen != fGeneration.load(std::memory_order_acquire))
         InitThreadSlot(td);
      return td;
   }
   /// Invalidate the per-thread data. Each thread rebuilds its own slot lazily on next access,
   /// so no cross-thread reach-in is needed.
   void ClearThreadData() const { fGeneration.fetch_add(1, std::memory_order_release); }
   /// No-op: this finder allocates its scratch state lazily for every calling thread.
   void CreateThreadData(Int_t) {}

protected:
   void InitThreadSlot(ThreadData_t &td) const;
   TGeoManager *GetOwnerManager() const { return fVolume ? fVolume->GetGeoManager() : nullptr; }
   TGeoMatrix *GetOwnerIdentity() const;
   void RegisterMatrix(TGeoMatrix *matrix) const;

   enum EGeoPatternFlags {
      kPatternReflected = BIT(14),
      kPatternSpacedOut = BIT(15)
   };
   Double_t fStep;      // division step length
   Double_t fStart;     // starting point on divided axis
   Double_t fEnd;       // ending point
   Int_t fNdivisions;   // number of divisions
   Int_t fDivIndex;     // index of first div. node
   TGeoVolume *fVolume; // volume to which applies

protected:
   TGeoPatternFinder(const TGeoPatternFinder &);
   TGeoPatternFinder &operator=(const TGeoPatternFinder &);

public:
   // constructors
   TGeoPatternFinder();
   TGeoPatternFinder(TGeoVolume *vol, Int_t ndiv);
   // destructor
   ~TGeoPatternFinder() override;
   // methods
   virtual TGeoMatrix *CreateMatrix() const = 0;
   virtual void cd(Int_t /*idiv*/) {}
   virtual TGeoNode *CdNext();
   virtual TGeoNode *FindNode(Double_t * /*point*/, const Double_t * /*dir*/ = nullptr) { return nullptr; }
   virtual Int_t GetByteCount() const { return 36; }
   Int_t GetCurrent(); //      {return fCurrent;}
   Int_t GetDivIndex() { return fDivIndex; }
   virtual Int_t GetDivAxis() { return 1; }
   virtual TGeoMatrix *GetMatrix(); //       {return fMatrix;}
   Int_t GetNdiv() const { return fNdivisions; }
   Int_t GetNext() const; //   {return fNextIndex;}
   TGeoNode *GetNodeOffset(Int_t idiv) { return fVolume->GetNode(fDivIndex + idiv); }
   Double_t GetStart() const { return fStart; }
   Double_t GetStep() const { return fStep; }
   Double_t GetEnd() const { return fEnd; }
   TGeoVolume *GetVolume() const { return fVolume; }
   virtual Bool_t IsOnBoundary(const Double_t * /*point*/) const { return kFALSE; }
   Bool_t IsReflected() const { return TObject::TestBit(kPatternReflected); }
   Bool_t IsSpacedOut() const { return TObject::TestBit(kPatternSpacedOut); }
   virtual TGeoPatternFinder *MakeCopy(Bool_t reflect = kFALSE) = 0;
   void Reflect(Bool_t flag = kTRUE) { TObject::SetBit(kPatternReflected, flag); }
   void SetDivIndex(Int_t index) { fDivIndex = index; }
   void SetNext(Int_t index); //     {fNextIndex = index;}
   void SetRange(Double_t start, Double_t step, Int_t ndivisions);
   void SetSpacedOut(Bool_t flag) { TObject::SetBit(kPatternSpacedOut, flag); }
   void SetVolume(TGeoVolume *vol) { fVolume = vol; }
   virtual void UpdateMatrix(Int_t, TGeoHMatrix &) const {}

   ClassDefOverride(TGeoPatternFinder, 4) // patterns to divide volumes
};

/// a X axis divison pattern
class TGeoTranslation;

class TGeoPatternX : public TGeoPatternFinder {
public:
   // constructors
   TGeoPatternX();
   TGeoPatternX(TGeoVolume *vol, Int_t ndivisions);
   TGeoPatternX(TGeoVolume *vol, Int_t ndivisions, Double_t step);
   TGeoPatternX(TGeoVolume *vol, Int_t ndivisions, Double_t start, Double_t end);
   TGeoPatternX(const TGeoPatternX &pf);
   TGeoPatternX &operator=(const TGeoPatternX &);

   // destructor
   ~TGeoPatternX() override;
   // methods
   TGeoMatrix *CreateMatrix() const override;
   void cd(Int_t idiv) override;
   TGeoNode *FindNode(Double_t *point, const Double_t *dir = nullptr) override;
   virtual Double_t FindNextBoundary(Double_t *point, Double_t *dir, Int_t &indnext);
   Int_t GetDivAxis() override { return 1; }
   Bool_t IsOnBoundary(const Double_t *point) const override;

   TGeoPatternFinder *MakeCopy(Bool_t reflect = kFALSE) override;
   void SavePrimitive(std::ostream &out, Option_t *option = "") override;
   void UpdateMatrix(Int_t idiv, TGeoHMatrix &matrix) const override;

   ClassDefOverride(TGeoPatternX, 1) // X division pattern
};

/// a Y axis divison pattern
class TGeoPatternY : public TGeoPatternFinder {
public:
   // constructors
   TGeoPatternY();
   TGeoPatternY(TGeoVolume *vol, Int_t ndivisions);
   TGeoPatternY(TGeoVolume *vol, Int_t ndivisions, Double_t step);
   TGeoPatternY(TGeoVolume *vol, Int_t ndivisions, Double_t start, Double_t end);
   TGeoPatternY(const TGeoPatternY &pf);
   TGeoPatternY &operator=(const TGeoPatternY &);
   // destructor
   ~TGeoPatternY() override;
   // methods
   TGeoMatrix *CreateMatrix() const override;
   void cd(Int_t idiv) override;
   TGeoNode *FindNode(Double_t *point, const Double_t *dir = nullptr) override;
   virtual Double_t FindNextBoundary(Double_t *point, Double_t *dir, Int_t &indnext);
   Int_t GetDivAxis() override { return 2; }
   Bool_t IsOnBoundary(const Double_t *point) const override;

   TGeoPatternFinder *MakeCopy(Bool_t reflect = kFALSE) override;
   void SavePrimitive(std::ostream &out, Option_t *option = "") override;
   void UpdateMatrix(Int_t idiv, TGeoHMatrix &matrix) const override;

   ClassDefOverride(TGeoPatternY, 1) // Y division pattern
};

/// a Z axis divison pattern
class TGeoPatternZ : public TGeoPatternFinder {
public:
   // constructors
   TGeoPatternZ();
   TGeoPatternZ(TGeoVolume *vol, Int_t ndivisions);
   TGeoPatternZ(TGeoVolume *vol, Int_t ndivisions, Double_t step);
   TGeoPatternZ(TGeoVolume *vol, Int_t ndivisions, Double_t start, Double_t end);
   TGeoPatternZ(const TGeoPatternZ &pf);
   TGeoPatternZ &operator=(const TGeoPatternZ &);
   // destructor
   ~TGeoPatternZ() override;
   // methods
   TGeoMatrix *CreateMatrix() const override;
   void cd(Int_t idiv) override;
   TGeoNode *FindNode(Double_t *point, const Double_t *dir = nullptr) override;
   virtual Double_t FindNextBoundary(Double_t *point, Double_t *dir, Int_t &indnext);
   Int_t GetDivAxis() override { return 3; }
   Bool_t IsOnBoundary(const Double_t *point) const override;

   TGeoPatternFinder *MakeCopy(Bool_t reflect = kFALSE) override;
   void SavePrimitive(std::ostream &out, Option_t *option = "") override;
   void UpdateMatrix(Int_t idiv, TGeoHMatrix &matrix) const override;

   ClassDefOverride(TGeoPatternZ, 1) // Z division pattern
};

/// a X axis divison pattern for PARA shapes
class TGeoPatternParaX : public TGeoPatternFinder {
public:
   // constructors
   TGeoPatternParaX();
   TGeoPatternParaX(TGeoVolume *vol, Int_t ndivisions);
   TGeoPatternParaX(TGeoVolume *vol, Int_t ndivisions, Double_t step);
   TGeoPatternParaX(TGeoVolume *vol, Int_t ndivisions, Double_t start, Double_t end);
   TGeoPatternParaX(const TGeoPatternParaX &pf);
   TGeoPatternParaX &operator=(const TGeoPatternParaX &);

   // destructor
   ~TGeoPatternParaX() override;
   // methods
   TGeoMatrix *CreateMatrix() const override;
   void cd(Int_t idiv) override;
   TGeoNode *FindNode(Double_t *point, const Double_t *dir = nullptr) override;
   Int_t GetDivAxis() override { return 1; }
   Bool_t IsOnBoundary(const Double_t *point) const override;

   TGeoPatternFinder *MakeCopy(Bool_t reflect = kFALSE) override;
   void SavePrimitive(std::ostream &out, Option_t *option = "") override;
   void UpdateMatrix(Int_t idiv, TGeoHMatrix &matrix) const override;

   ClassDefOverride(TGeoPatternParaX, 1) // Para X division pattern
};

/// a Y axis divison pattern for PARA shapes
class TGeoPatternParaY : public TGeoPatternFinder {
private:
   // data members
   Double_t fTxy = 0.; // tangent of alpha
public:
   // constructors
   TGeoPatternParaY();
   TGeoPatternParaY(TGeoVolume *vol, Int_t ndivisions);
   TGeoPatternParaY(TGeoVolume *vol, Int_t ndivisions, Double_t step);
   TGeoPatternParaY(TGeoVolume *vol, Int_t ndivisions, Double_t start, Double_t end);
   TGeoPatternParaY(const TGeoPatternParaY &pf);
   TGeoPatternParaY &operator=(const TGeoPatternParaY &);

   // destructor
   ~TGeoPatternParaY() override;
   // methods
   TGeoMatrix *CreateMatrix() const override;
   void cd(Int_t idiv) override;
   TGeoNode *FindNode(Double_t *point, const Double_t *dir = nullptr) override;
   Int_t GetDivAxis() override { return 2; }
   Bool_t IsOnBoundary(const Double_t *point) const override;

   TGeoPatternFinder *MakeCopy(Bool_t reflect = kFALSE) override;
   void SavePrimitive(std::ostream &out, Option_t *option = "") override;
   void UpdateMatrix(Int_t idiv, TGeoHMatrix &matrix) const override;

   ClassDefOverride(TGeoPatternParaY, 1) // Para Y division pattern
};

/// a Z axis divison pattern for PARA shapes
class TGeoPatternParaZ : public TGeoPatternFinder {
private:
   // data members
   Double_t fTxz = 0.; // tangent of alpha xz
   Double_t fTyz = 0.; // tangent of alpha yz
public:
   // constructors
   TGeoPatternParaZ();
   TGeoPatternParaZ(TGeoVolume *vol, Int_t ndivisions);
   TGeoPatternParaZ(TGeoVolume *vol, Int_t ndivisions, Double_t step);
   TGeoPatternParaZ(TGeoVolume *vol, Int_t ndivisions, Double_t start, Double_t end);
   TGeoPatternParaZ(const TGeoPatternParaZ &pf);
   TGeoPatternParaZ &operator=(const TGeoPatternParaZ &);

   // destructor
   ~TGeoPatternParaZ() override;
   // methods
   TGeoMatrix *CreateMatrix() const override;
   void cd(Int_t idiv) override;
   TGeoNode *FindNode(Double_t *point, const Double_t *dir = nullptr) override;
   Int_t GetDivAxis() override { return 3; }
   Bool_t IsOnBoundary(const Double_t *point) const override;

   TGeoPatternFinder *MakeCopy(Bool_t reflect = kFALSE) override;
   void SavePrimitive(std::ostream &out, Option_t *option = "") override;
   void UpdateMatrix(Int_t idiv, TGeoHMatrix &matrix) const override;

   ClassDefOverride(TGeoPatternParaZ, 1) // Para Z division pattern
};

/// a Z axis divison pattern for TRAP or GTRA shapes
class TGeoPatternTrapZ : public TGeoPatternFinder {
private:
   // data members
   Double_t fTxz = 0.; // tangent of alpha xz
   Double_t fTyz = 0.; // tangent of alpha yz
public:
   // constructors
   TGeoPatternTrapZ();
   TGeoPatternTrapZ(TGeoVolume *vol, Int_t ndivisions);
   TGeoPatternTrapZ(TGeoVolume *vol, Int_t ndivisions, Double_t step);
   TGeoPatternTrapZ(TGeoVolume *vol, Int_t ndivisions, Double_t start, Double_t end);
   TGeoPatternTrapZ(const TGeoPatternTrapZ &pf);
   TGeoPatternTrapZ &operator=(const TGeoPatternTrapZ &);

   // destructor
   ~TGeoPatternTrapZ() override;
   // methods
   TGeoMatrix *CreateMatrix() const override;
   Double_t GetTxz() const { return fTxz; }
   Double_t GetTyz() const { return fTyz; }
   void cd(Int_t idiv) override;
   TGeoNode *FindNode(Double_t *point, const Double_t *dir = nullptr) override;
   Int_t GetDivAxis() override { return 3; }
   Bool_t IsOnBoundary(const Double_t *point) const override;

   TGeoPatternFinder *MakeCopy(Bool_t reflect = kFALSE) override;
   void SavePrimitive(std::ostream &out, Option_t *option = "") override;
   void UpdateMatrix(Int_t idiv, TGeoHMatrix &matrix) const override;

   ClassDefOverride(TGeoPatternTrapZ, 1) // Trap od Gtra Z division pattern
};

/// a cylindrical R divison pattern
class TGeoPatternCylR : public TGeoPatternFinder {
public:
   // constructors
   TGeoPatternCylR();
   TGeoPatternCylR(TGeoVolume *vol, Int_t ndivisions);
   TGeoPatternCylR(TGeoVolume *vol, Int_t ndivisions, Double_t step);
   TGeoPatternCylR(TGeoVolume *vol, Int_t ndivisions, Double_t start, Double_t end);
   TGeoPatternCylR(const TGeoPatternCylR &pf);
   TGeoPatternCylR &operator=(const TGeoPatternCylR &);
   // destructor
   ~TGeoPatternCylR() override;
   // methods
   TGeoMatrix *CreateMatrix() const override;
   void cd(Int_t idiv) override;
   TGeoNode *FindNode(Double_t *point, const Double_t *dir = nullptr) override;
   Int_t GetDivAxis() override { return 1; }
   Bool_t IsOnBoundary(const Double_t *point) const override;

   TGeoPatternFinder *MakeCopy(Bool_t reflect = kFALSE) override;
   void SavePrimitive(std::ostream &out, Option_t *option = "") override;
   void UpdateMatrix(Int_t idiv, TGeoHMatrix &matrix) const override;

   ClassDefOverride(TGeoPatternCylR, 1) // Cylindrical R division pattern
};

/// a cylindrical phi divison pattern
class TGeoPatternCylPhi : public TGeoPatternFinder {
private:
   // data members
   Double_t *fSinCos = nullptr; ///<![2*fNdivisions] table of sines/cosines

protected:
   TGeoPatternCylPhi(const TGeoPatternCylPhi &pfc) : TGeoPatternFinder(pfc), fSinCos(pfc.fSinCos)
   {
      CreateThreadData(1);
   }
   TGeoPatternCylPhi &operator=(const TGeoPatternCylPhi &pfc)
   {
      if (this != &pfc) {
         TGeoPatternFinder::operator=(pfc);
         fSinCos = pfc.fSinCos;
         CreateThreadData(1);
      }
      return *this;
   }

public:
   // constructors
   TGeoPatternCylPhi();
   TGeoPatternCylPhi(TGeoVolume *vol, Int_t ndivisions);
   TGeoPatternCylPhi(TGeoVolume *vol, Int_t ndivisions, Double_t step);
   TGeoPatternCylPhi(TGeoVolume *vol, Int_t ndivisions, Double_t start, Double_t end);
   // destructor
   ~TGeoPatternCylPhi() override;
   // methods
   TGeoMatrix *CreateMatrix() const override;
   void cd(Int_t idiv) override;
   TGeoNode *FindNode(Double_t *point, const Double_t *dir = nullptr) override;
   Int_t GetDivAxis() override { return 2; }
   Bool_t IsOnBoundary(const Double_t *point) const override;

   TGeoPatternFinder *MakeCopy(Bool_t reflect = kFALSE) override;
   void SavePrimitive(std::ostream &out, Option_t *option = "") override;
   void UpdateMatrix(Int_t idiv, TGeoHMatrix &matrix) const override;

   ClassDefOverride(TGeoPatternCylPhi, 1) // Cylindrical phi division pattern
};

/// a spherical R divison pattern
class TGeoPatternSphR : public TGeoPatternFinder {
public:
   // constructors
   TGeoPatternSphR();
   TGeoPatternSphR(TGeoVolume *vol, Int_t ndivisions);
   TGeoPatternSphR(TGeoVolume *vol, Int_t ndivisions, Double_t step);
   TGeoPatternSphR(TGeoVolume *vol, Int_t ndivisions, Double_t start, Double_t end);
   TGeoPatternSphR(const TGeoPatternSphR &pf);
   TGeoPatternSphR &operator=(const TGeoPatternSphR &);
   // destructor
   ~TGeoPatternSphR() override;
   // methods
   TGeoMatrix *CreateMatrix() const override;
   void cd(Int_t idiv) override;
   TGeoNode *FindNode(Double_t *point, const Double_t *dir = nullptr) override;
   Int_t GetDivAxis() override { return 1; }

   TGeoPatternFinder *MakeCopy(Bool_t reflect = kFALSE) override;
   void SavePrimitive(std::ostream &out, Option_t *option = "") override;
   void UpdateMatrix(Int_t idiv, TGeoHMatrix &matrix) const override;

   ClassDefOverride(TGeoPatternSphR, 1) // spherical R division pattern
};

/// a spherical theta divison pattern
class TGeoPatternSphTheta : public TGeoPatternFinder {
public:
   // constructors
   TGeoPatternSphTheta();
   TGeoPatternSphTheta(TGeoVolume *vol, Int_t ndivisions);
   TGeoPatternSphTheta(TGeoVolume *vol, Int_t ndivisions, Double_t step);
   TGeoPatternSphTheta(TGeoVolume *vol, Int_t ndivisions, Double_t start, Double_t end);
   TGeoPatternSphTheta(const TGeoPatternSphTheta &pf);
   TGeoPatternSphTheta &operator=(const TGeoPatternSphTheta &);
   // destructor
   ~TGeoPatternSphTheta() override;
   // methods
   TGeoMatrix *CreateMatrix() const override;
   void cd(Int_t idiv) override;
   TGeoNode *FindNode(Double_t *point, const Double_t *dir = nullptr) override;
   Int_t GetDivAxis() override { return 3; }

   TGeoPatternFinder *MakeCopy(Bool_t reflect = kFALSE) override;
   void SavePrimitive(std::ostream &out, Option_t *option = "") override;
   void UpdateMatrix(Int_t idiv, TGeoHMatrix &matrix) const override;

   ClassDefOverride(TGeoPatternSphTheta, 1) // spherical theta division pattern
};

/// a spherical phi divison pattern
class TGeoPatternSphPhi : public TGeoPatternFinder {
private:
   Double_t *fSinCos = nullptr; ///<! Sincos table

protected:
   TGeoPatternSphPhi(const TGeoPatternSphPhi &pfc) = delete;
   TGeoPatternSphPhi &operator=(const TGeoPatternSphPhi &pfc) = delete;
   Double_t *CreateSinCos();

public:
   // constructors
   TGeoPatternSphPhi();
   TGeoPatternSphPhi(TGeoVolume *vol, Int_t ndivisions);
   TGeoPatternSphPhi(TGeoVolume *vol, Int_t ndivisions, Double_t step);
   TGeoPatternSphPhi(TGeoVolume *vol, Int_t ndivisions, Double_t start, Double_t end);
   // destructor
   ~TGeoPatternSphPhi() override;
   // methods
   TGeoMatrix *CreateMatrix() const override;
   void cd(Int_t idiv) override;
   TGeoNode *FindNode(Double_t *point, const Double_t *dir = nullptr) override;
   Int_t GetDivAxis() override { return 2; }
   Bool_t IsOnBoundary(const Double_t *point) const override;

   TGeoPatternFinder *MakeCopy(Bool_t reflect = kFALSE) override;
   void SavePrimitive(std::ostream &out, Option_t *option = "") override;
   void UpdateMatrix(Int_t idiv, TGeoHMatrix &matrix) const override;

   ClassDefOverride(TGeoPatternSphPhi, 1) // Spherical phi division pattern
};

/// a divison pattern specialized for honeycombs
class TGeoPatternHoneycomb : public TGeoPatternFinder {
private:
   // data members
   Int_t fNrows;       // number of rows
   Int_t fAxisOnRows;  // axis along each row
   Int_t *fNdivisions; // [fNrows] number of divisions for each row
   Double_t *fStart;   // [fNrows] starting points for each row

protected:
   TGeoPatternHoneycomb(const TGeoPatternHoneycomb &);
   TGeoPatternHoneycomb &operator=(const TGeoPatternHoneycomb &);

public:
   // constructors
   TGeoPatternHoneycomb();
   TGeoPatternHoneycomb(TGeoVolume *vol, Int_t nrows);
   // destructor
   ~TGeoPatternHoneycomb() override;
   // methods
   TGeoPatternFinder *MakeCopy(Bool_t) override { return nullptr; }
   TGeoMatrix *CreateMatrix() const override;
   void cd(Int_t idiv) override;
   TGeoNode *FindNode(Double_t *point, const Double_t *dir = nullptr) override;
   void UpdateMatrix(Int_t idiv, TGeoHMatrix &matrix) const override;

   ClassDefOverride(TGeoPatternHoneycomb, 1) // pattern for honeycomb divisions
};

#endif
