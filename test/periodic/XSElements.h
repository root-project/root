/*
 * $Header$
 * $Log$
 */

#ifndef __XSELEMENTS_H
#define __XSELEMENTS_H

#include <TObject.h>
#include <TObjArray.h>

enum Element_State {
   Element_GroundState,
   Element_Metastable,
   Element_SecondMetastable
};

/* =================== XSElement ===================== */
class XSElement : public TObject
{
protected:
   Int_t   z;
   char   *name;
   char   *symbol;

   char   *atomic_weight;
   char   *density;
   char   *melting_point;
   char   *boiling_point;
   char   *oxidation_states;

   Int_t   ni;
   char   **isotope;
   char   **isotope_info;
   Bool_t   *isotope_stable;

public:
   XSElement();
   ~XSElement();

   inline char*    Name()      const { return name; }
   inline char*   Mnemonic()   const { return symbol; }
   inline char*   Symbol()   const { return symbol; }
   inline char*   AtomicWeight()   const { return atomic_weight; }
   inline char*   Density()   const { return density; }
   inline char*   MeltingPt()   const { return melting_point; }
   inline char*   BoilingPt()   const { return boiling_point; }
   inline char*   Oxidation()   const { return oxidation_states; }
   inline Int_t   Isotopes()   const { return ni; }
   inline char*   Isotope(int i)   const { return isotope[i]; }
   const char*    IsotopeInfo(const char *isotope);
   inline char*   IsotopeInfo(int i)
   const { return isotope_info[i]; }
   inline Bool_t  IsStable(int i)   const { return isotope_stable[i]; }

   void   Read(FILE *f);
   Int_t   Read(const char *name) { return TObject::Read(name); }
protected:
   char*   ReadLine(FILE *f);
   //ClassDef(XSElement,1)
}; // XSElement

/* =================== XSElements ===================== */
class XSElements : public TObject
{
protected:
   UInt_t      NElements;
   TObjArray   *elements;

public:
   XSElements( const char *filename );
   ~XSElements();

   inline UInt_t   GetSize()   const   { return NElements; }

   inline char*   Name(Int_t Z)
   { return ((XSElement*)((*elements)[Z-1]))->Name(); }

   inline char*   Mnemonic(Int_t Z)
   { return ((XSElement*)((*elements)[Z-1]))->Mnemonic(); }

   inline XSElement   *Elem(Int_t Z)
   { return ((XSElement*)((*elements)[Z-1])); }

   // Search for element either by name or mnemonic
   UInt_t      Find(const char *str);

   //ClassDef(XSElements,1)
}; // XSElements

#endif
