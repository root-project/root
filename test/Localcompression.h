#ifndef ROOT_Localcompression
#define ROOT_Localcompression

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Local Compression                                                    //
//                                                                      //
// Description of the event and track parameters                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TObject.h"
#include "TClonesArray.h"
#include "TRefArray.h"
#include "TRef.h"
#include "TH1.h"
#include "TBits.h"
#include "TMath.h"

#define LARGESIZE 1000000
#define SMALLSIZE 1000

class TLarge : public TObject {

private:
   Int_t         fSize;
   Float_t      *fLarge; //[fSize]

public:
   TLarge(Int_t size = LARGESIZE); 
   TLarge(const TLarge& large);
   virtual ~TLarge();
   TLarge &operator=(const TLarge &large);

   void          Clear(Option_t *option ="");
   void          Build();
   Int_t         GetSize() const { return fSize; }
   Float_t      *GetLarge() const { return fLarge; }

   ClassDef(TLarge,1)
};

class TSmall : public TObject {

private:
   Int_t         fSize;
   Float_t      *fSmall; //[fSize]

public:
   TSmall(Int_t size = SMALLSIZE); 
   TSmall(const TSmall& small);
   virtual ~TSmall();
   TSmall &operator=(const TSmall &small);

   void          Clear(Option_t *option ="");
   void          Build();
   Int_t         GetSize() const { return fSize; }
   Float_t      *GetSmall() const { return fSmall; }

   ClassDef(TSmall,1)
};

class TInt : public TObject {

private:
   Int_t         fSize;
   Int_t        *fInt; //[fSize]

public:
   TInt(Int_t size = 6); 
   TInt(const TInt& aint);
   virtual ~TInt();
   TInt &operator=(const TInt &aint);

   void          Clear(Option_t *option ="");
   void          Build();
   Int_t         GetSize() const { return fSize; }
   Int_t        *GetInt() const { return fInt; }

   ClassDef(TInt,1)
};

#endif
