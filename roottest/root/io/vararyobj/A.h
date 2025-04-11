// A.h

#ifndef A_HDR
#define A_HDR

#include "B.h"
#include "TObject.h"

class A : public TObject {
private:
  static const int fM = 5;
  int           fN;                  //     Varying-size array counter.
  B             *fTable;             //[fN] Pointer to a varying-size array of B objects.
  B            **fPtrTable;          //[fN] Pointer to a varying-size array of pointers to B objects.
  B             *fFixedTable[fM];    //[fN] A fixed-size array of pointers to varying size arrays of B objects.
  B            **fFixedPtrTable[fM]; //[fN] A fixed-size array of pointers to varying size arrays of pointers to B objects.

public:
  //
  // Special Member Functions
  //
  A();
  ~A();
  A(const A&);
  A& operator=(const A&);

public:
  //
  // Accessors
  //
  int            GetN() const       { return fN; }             // Get varying-size array counter.
  B             *GetTable()         { return fTable; }         // Get pointer to varying-size array of B objects.
  B            **GetPtrTable()      { return fPtrTable; }      // Get pointer to varying-size array of pointers to B objects.
  B            **GetFixedTable()    { return fFixedTable; }    // Get a fixed-size array of pointers to varying size arrays of B objects.
  B           ***GetFixedPtrTable() { return fFixedPtrTable; } // Get a fixed-size array of pointers to varying size arrays of pointers to B objects.
  void           repr() const;

  //
  // Modifiers
  //
  void Init();

public:
  ClassDef(A,1);
};

#endif // A_HDR
