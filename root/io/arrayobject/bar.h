#ifndef BAR_H
#define BAR_H

#include "TROOT.h"
#include "TClonesArray.h"
#include "foo.h"
#include <iostream>

class bar: public TObject {
public:
  bar() {
    v[0] = 0;
    v[1] = 0;
    v[2] = 0;
    fp[0] = new foo;
    fp[1] = new foo;


    fop[0] = new foobj;
    fop[1] = new foobj;

    fov = new foobj[2];
    vop = 2;
    fvop = new foobj[vop];

    fv = new foo[2];
    vp = 2;
    fvp = new foo[vp];

  }
  ~bar() {}

  void print() {
    std::cerr

      <<  "fop[0].i " << fop[0]->i 
      << " fop[1].i " << fop[1]->i << std::endl

      << "fo [0].i "  << fo [0].i 
      << " fo [1].i " << fo [1].i << std::endl

      << "fp [0].i "  << fp [0]->i 
      << " fp [1].i " << fp [1]->i << std::endl

      << "f  [0].i "  << f  [0].i 
      << " f  [1].i " << f  [1].i << std::endl
      

      <<  "fov[0].i " << fov[0].i 
      << " fov[1].i " << fov[1].i << std::endl

      << "fvop[0].i "  << fvop[0].i 
      << " fvop[1].i " << fvop[1].i << std::endl

      << "fv [0].i "  << fv [0].i 
      << " fv [1].i " << fv [1].i << std::endl

      << "fvp[0].i "  << fvp[0].i 
      << " fvp[1].i " << fvp[1].i << std::endl
      
      << std::endl;
  }

  Int_t v[3];
#ifdef _2_CINT__
  foo  fo [2]; //[2]
#else
  foobj *fop[2]; //
  foobj  fo [2]; //
  foo   *fp [2]; //
  foo    f  [2]; //

  foobj *fov;    //! fixed size is not supported [2]
  int    vop;     
  foobj *fvop;   //[vop]

  foo   *fv;     //! fixed size is not supported [2]
  int    vp;     
  foo  *fvp;     //[vp]
#endif
  
  int Getfp_is() {
    return fp[0]->i + fp[1]->i;
  };

  ClassDef(bar,2)
};
#endif
