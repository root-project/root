// @(#)root/foam:$Id$
// Author: S. Jadach <mailto:Stanislaw.jadach@ifj.edu.pl>, P.Sawicki <mailto:Pawel.Sawicki@ifj.edu.pl>

#ifndef ROOT_TFoamVect
#define ROOT_TFoamVect

////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Auxiliary class TFoamVect of n-dimensional vector, with dynamic allocation //
// used for the cartesian geometry of the TFoam cells                         //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif


///////////////////////////////////////////////////////////////////////////////
class TFoamVect : public TObject {
   // constructor
private:
   Int_t       fDim;                     // Dimension
   Double_t   *fCoords;                  // [fDim] Coordinates
public:
   TFoamVect();                          // Constructor
   TFoamVect(Int_t);                     // USER Constructor
   TFoamVect(const TFoamVect &);         // Copy constructor
   virtual ~TFoamVect();                 // Destructor
//////////////////////////////////////////////////////////////////////////////
//                     Overloading operators                                //
//////////////////////////////////////////////////////////////////////////////
   TFoamVect& operator =(const TFoamVect&);  // = operator; Substitution
   Double_t &operator[](Int_t);              // [] provides POINTER to coordinate
   TFoamVect& operator =(Double_t []);       // LOAD IN entire double vector
   TFoamVect& operator =(Double_t);          // LOAD IN double number
//////////////////////////   OTHER METHODS    //////////////////////////////////
   TFoamVect& operator+=(const  TFoamVect&); // +=; add vector u+=v  (FAST)
   TFoamVect& operator-=(const  TFoamVect&); // +=; add vector u+=v  (FAST)
   TFoamVect& operator*=(const Double_t&);   // *=; mult. by scalar v*=x (FAST)
   TFoamVect  operator+( const  TFoamVect&); // +;  u=v+s, NEVER USE IT, SLOW!!!
   TFoamVect  operator-( const  TFoamVect&); // -;  u=v-s, NEVER USE IT, SLOW!!!
   void Print(Option_t *option) const;   // Prints vector
   Int_t    GetDim() const { return fDim; }  // Returns dimension
   Double_t GetCoord(Int_t i) const {return fCoords[i];};   // Returns coordinate

   ClassDef(TFoamVect,1) //n-dimensional vector with dynamical allocation
};

#endif

