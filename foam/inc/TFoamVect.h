
// $Id: TFoamVect.h,v 1.2 2005/04/04 10:59:34 psawicki Exp $

#ifndef ROOT_TFoamVect
#define ROOT_TFoamVect

////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Auxiliary class TFoamVect of n-dimensional vector, with dynamic allocation //
// used for the cartesian geometry of the TFoam cells                         //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include "TROOT.h"
#include "TNamed.h"
#include "TObject.h"

///////////////////////////////////////////////////////////////////////////////
class TFoamVect : public TObject {
  // constructor
  private:
    static const Int_t fchat =0;          // chatability (for debug)
  private:
    Int_t       fDim;                     // Dimension
    Double_t   *fCoords;                  // [fDim] Coordinates
    TFoamVect  *fNext;                    // pointer for tree construction
    TFoamVect  *fPrev;                    // pointer for tree construction
  public:
    TFoamVect();                          // Constructor
    TFoamVect(const Int_t );              // USER Constructor
    TFoamVect(const TFoamVect &);         // Copy constructor
    virtual ~TFoamVect();                 // Destructor
//////////////////////////////////////////////////////////////////////////////
//                     Overloading operators                                //
//////////////////////////////////////////////////////////////////////////////
    TFoamVect& operator =(const TFoamVect&);  // = operator; Substitution
    Double_t &operator[](Int_t);              // [] provides POINTER to coordinate
    TFoamVect& operator =(Double_t []);       // LOAD IN entire double vector
    TFoamVect& operator =(Double_t);          // LOAD IN double numer
//////////////////////////   OTHER METHODS    //////////////////////////////////
    TFoamVect& operator+=(const  TFoamVect&); // +=; add vector u+=v  (FAST)
    TFoamVect& operator-=(const  TFoamVect&); // +=; add vector u+=v  (FAST)
    TFoamVect& operator*=(const Double_t&);   // *=; mult. by scalar v*=x (FAST)
    TFoamVect  operator+( const  TFoamVect&); // +;  u=v+s, NEVER USE IT, SLOW!!!
    TFoamVect  operator-( const  TFoamVect&); // -;  u=v-s, NEVER USE IT, SLOW!!!
    void PrintCoord(void);                    // Prints vector
    void PrintList(void);                     // Prints vector and the following linked list
    const int &GetDim(void);                  // Returns dimension
    Double_t GetCoord(Int_t i){return fCoords[i];};   // Returns coordinate
/////////////////////////////////////////////////////////////////////////////
    ClassDef(TFoamVect,1); //n-dimensional vector with dynamical allocation
};
/////////////////////////////////////////////////////////////////////////////
#endif

