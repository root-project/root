/*
  File: roottest/python/basic/DataTypes.C
  Author: Wim Lavrijsen@lbl.gov
  Created: 05/11/05
  Last: 05/16/05
*/

const int N = 5;

struct Pod {
   Int_t    fInt;
   Double_t fDouble;
};

class ClassWithData {
public:
   ClassWithData() : fOwnsArrays( false )
   {
      fBool   = kFALSE;
      fChar   = 'a';
      fUChar  = 'c';
      fShort  = -11;
      fUShort =  11u;
      fInt    = -22;
      fUInt   =  22u;
      fLong   = -33l;
      fULong  =  33ul;
      fFloat  = -44.f;
      fDouble = -55.;

      fShortArray2  = new Short_t[N];
      fUShortArray2 = new UShort_t[N];
      fIntArray2    = new Int_t[N];
      fUIntArray2   = new UInt_t[N];
      fLongArray2   = new Long_t[N];
      fULongArray2  = new ULong_t[N];

      fFloatArray2  = new Float_t[N];
      fDoubleArray2 = new Double_t[N];

      for ( int i = 0; i < N; ++i ) {
         fShortArray[i]   =  -1*i;
         fShortArray2[i]  =  -2*i;
         fUShortArray[i]  =   3u*i;
         fUShortArray2[i] =   4u*i;
         fIntArray[i]     =  -5*i;
         fIntArray2[i]    =  -6*i;
         fUIntArray[i]    =   7u*i;
         fUIntArray2[i]   =   8u*i;
         fLongArray[i]    =  -9l*i;
         fLongArray2[i]   = -10l*i;
         fULongArray[i]   =  11ul*i;
         fULongArray2[i]  =  12ul*i;

         fFloatArray[i]   = -13.f*i;
         fFloatArray2[i]  = -14.f*i;
         fDoubleArray[i]  = -15.*i;
         fDoubleArray2[i] = -16.*i;
      }

      fOwnsArrays = true;

      fPod.fInt    = 888;
      fPod.fDouble = 3.14;
   };

   ~ClassWithData()
   {
      DestroyArrays();
   }

   void DestroyArrays() {
      if ( fOwnsArrays == true ) {
         delete[] fShortArray2;
         delete[] fUShortArray2;
         delete[] fIntArray2;
         delete[] fUIntArray2;
         delete[] fLongArray2;
         delete[] fULongArray2;

         delete[] fFloatArray2;
         delete[] fDoubleArray2;

         fOwnsArrays = false;
      }
   }

// getters
   Bool_t   GetBool()   { return fBool; }
   Char_t   GetChar()   { return fChar; }
   UChar_t  GetUChar()  { return fUChar; }
   Short_t  GetShort()  { return fShort; }
   UShort_t GetUShort() { return fUShort; }
   Int_t    GetInt()    { return fInt; }
   UInt_t   GetUInt()   { return fUInt; }
   Long_t   GetLong()   { return fLong; }
   ULong_t  GetULong()  { return fULong; }
   Float_t  GetFloat()  { return fFloat; }
   Double_t GetDouble() { return fDouble; }

   Short_t*  GetShortArray()   { return fShortArray; }
   Short_t*  GetShortArray2()  { return fShortArray2; }
   UShort_t* GetUShortArray()  { return fUShortArray; }
   UShort_t* GetUShortArray2() { return fUShortArray2; }
   Int_t*    GetIntArray()     { return fIntArray; }
   Int_t*    GetIntArray2()    { return fIntArray2; }
   UInt_t*   GetUIntArray()    { return fUIntArray; }
   UInt_t*   GetUIntArray2()   { return fUIntArray2; }
   Long_t*   GetLongArray()    { return fLongArray; }
   Long_t*   GetLongArray2()   { return fLongArray2; }
   ULong_t*  GetULongArray()   { return fULongArray; }
   ULong_t*  GetULongArray2()  { return fULongArray2; }

   Float_t*  GetFloatArray()   { return fFloatArray; }
   Float_t*  GetFloatArray2()  { return fFloatArray2; }
   Double_t* GetDoubleArray()  { return fDoubleArray; }
   Double_t* GetDoubleArray2() { return fDoubleArray2; }

// setters
   void SetBool( Bool_t b )      { fBool   = b;  }
   void SetChar( Char_t c )      { fChar   = c;  }
   void SetUChar( UChar_t uc )   { fUChar  = uc; }
   void SetShort( Short_t s )    { fShort  = s;  }
   void SetUShort( UShort_t us ) { fUShort = us; }
   void SetInt( Int_t i )        { fInt    = i;  }
   void SetUInt( UInt_t ui )     { fUInt   = ui; }
   void SetLong( Long_t l )      { fLong   = l;  }
   void SetULong( ULong_t ul )   { fULong  = ul; }
   void SetFloat( Float_t f )    { fFloat  = f;  }
   void SetDouble( Double_t d )  { fDouble = d;  }

public:
// basic types
   Bool_t   fBool;
   Char_t   fChar;
   UChar_t  fUChar;
   Short_t  fShort;
   UShort_t fUShort;
   Int_t    fInt;
   UInt_t   fUInt;
   Long_t   fLong;
   ULong_t  fULong;
   Float_t  fFloat;
   Double_t fDouble;

// array types
   Short_t   fShortArray[N];
   Short_t*  fShortArray2;
   UShort_t  fUShortArray[N];
   UShort_t* fUShortArray2;
   Int_t     fIntArray[N];
   Int_t*    fIntArray2;
   UInt_t    fUIntArray[N];
   UInt_t*   fUIntArray2;
   Long_t    fLongArray[N];
   Long_t*   fLongArray2;
   ULong_t   fULongArray[N];
   ULong_t*  fULongArray2;

   Float_t   fFloatArray[N];
   Float_t*  fFloatArray2;
   Double_t  fDoubleArray[N];
   Double_t* fDoubleArray2;

// object types
   Pod fPod;

public:
   static Char_t   sChar;
   static UChar_t  sUChar;
   static Short_t  sShort;
   static UShort_t sUShort;
   static Int_t    sInt;
   static UInt_t   sUInt;
   static Long_t   sLong;
   static ULong_t  sULong;
   static Float_t  sFloat;
   static Double_t sDouble;

private:
   bool fOwnsArrays;
};

Char_t   ClassWithData::sChar   = 's';
UChar_t  ClassWithData::sUChar  = 'u';
Short_t  ClassWithData::sShort  = -101;
UShort_t ClassWithData::sUShort =  255u;
Int_t    ClassWithData::sInt    = -202;
UInt_t   ClassWithData::sUInt   =  202u;
Long_t   ClassWithData::sLong   = -303l;
ULong_t  ClassWithData::sULong  =  303ul;
Float_t  ClassWithData::sFloat  = -404.f;
Double_t ClassWithData::sDouble = -505.;

long GetPodAddress( ClassWithData& c )
{
   return (long)&c.fPod;
}

long GetIntAddress( ClassWithData& c )
{
   return (long)&c.fPod.fInt;
}

long GetDoubleAddress( ClassWithData& c )
{
   return (long)&c.fPod.fDouble;
}
