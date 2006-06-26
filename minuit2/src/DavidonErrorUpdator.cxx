// @(#)root/minuit2:$Name:  $:$Id: DavidonErrorUpdator.cxx,v 1.1 2005/11/29 14:43:31 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/DavidonErrorUpdator.h"
#include "Minuit2/MinimumState.h"
#include "Minuit2/LaSum.h"
#include "Minuit2/LaProd.h"

namespace ROOT {

   namespace Minuit2 {


double inner_product(const LAVector&, const LAVector&);
double similarity(const LAVector&, const LASymMatrix&);
double sum_of_elements(const LASymMatrix&);

MinimumError DavidonErrorUpdator::Update(const MinimumState& s0, 
                                         const MinimumParameters& p1,
                                         const FunctionGradient& g1) const {

   // update of the covarianze matrix (Davidon formula, see Tutorial, par. 4.8 pag 26)
   // in case of delgam > gvg (PHI > 1) use rank one formula 
   // see  par 4.10 pag 30

   const MnAlgebraicSymMatrix& V0 = s0.Error().InvHessian();
   MnAlgebraicVector dx = p1.Vec() - s0.Vec();
   MnAlgebraicVector dg = g1.Vec() - s0.Gradient().Vec();
  
   double delgam = inner_product(dx, dg);
   double gvg = similarity(dg, V0);

   //   std::cout<<"delgam= "<<delgam<<" gvg= "<<gvg<<std::endl;
   MnAlgebraicVector vg = V0*dg;

   MnAlgebraicSymMatrix Vupd = Outer_product(dx)/delgam - Outer_product(vg)/gvg;

   if(delgam > gvg) {
      // use rank 1 formula
      Vupd += gvg*Outer_product(MnAlgebraicVector(dx/delgam - vg/gvg));
   }

   double sum_upd = sum_of_elements(Vupd);
   Vupd += V0;
  
   double dcov = 0.5*(s0.Error().Dcovar() + sum_upd/sum_of_elements(Vupd));
  
   return MinimumError(Vupd, dcov);
}

/*
MinimumError DavidonErrorUpdator::Update(const MinimumState& s0, 
					 const MinimumParameters& p1,
					 const FunctionGradient& g1) const {

  const MnAlgebraicSymMatrix& V0 = s0.Error().InvHessian();
  MnAlgebraicVector dx = p1.Vec() - s0.Vec();
  MnAlgebraicVector dg = g1.Vec() - s0.Gradient().Vec();
  
  double delgam = inner_product(dx, dg);
  double gvg = similarity(dg, V0);

//   std::cout<<"delgam= "<<delgam<<" gvg= "<<gvg<<std::endl;
  MnAlgebraicVector vg = V0*dg;
//   MnAlgebraicSymMatrix Vupd(V0.Nrow());

//   MnAlgebraicSymMatrix dd = ( 1./delgam )*outer_product(dx);
//   dd *= ( 1./delgam );
//   MnAlgebraicSymMatrix VggV = ( 1./gvg )*outer_product(vg);
//   VggV *= ( 1./gvg );
//   Vupd = dd - VggV;
//   MnAlgebraicSymMatrix Vupd = ( 1./delgam )*outer_product(dx) - ( 1./gvg )*outer_product(vg);
  MnAlgebraicSymMatrix Vupd = Outer_product(dx)/delgam - Outer_product(vg)/gvg;
  
  if(delgam > gvg) {
//     dx *= ( 1./delgam );
//     vg *= ( 1./gvg );
//     MnAlgebraicVector flnu = dx - vg;
//     MnAlgebraicSymMatrix tmp = Outer_product(flnu);
//     tmp *= gvg;
//     Vupd = Vupd + tmp;
    Vupd += gvg*outer_product(dx/delgam - vg/gvg);
  }

//   
//     MnAlgebraicSymMatrix dd = Outer_product(dx);
//     dd *= ( 1./delgam );
//     MnAlgebraicSymMatrix VggV = Outer_product(vg);
//     VggV *= ( 1./gvg );
//     Vupd = dd - VggV;
//   
//     
//   double phi = delgam/(delgam - gvg);

//   MnAlgebraicSymMatrix Vupd(V0.Nrow());
//   if(phi < 0) {
//     // rank-2 Update
//     MnAlgebraicSymMatrix dd = Outer_product(dx);
//     dd *= ( 1./delgam );
//     MnAlgebraicSymMatrix VggV = Outer_product(vg);
//     VggV *= ( 1./gvg );
//     Vupd = dd - VggV;
//   }
//   if(phi > 1) {
//     // rank-1 Update
//     MnAlgebraicVector tmp = dx - vg;
//     Vupd = Outer_product(tmp);
//     Vupd *= ( 1./(delgam - gvg) );
//   }
//     

//     
//   if(delgam > gvg) {
//     // rank-1 Update
//     MnAlgebraicVector tmp = dx - vg;
//     Vupd = Outer_product(tmp);
//     Vupd *= ( 1./(delgam - gvg) );
//   } else { 
//     // rank-2 Update
//     MnAlgebraicSymMatrix dd = Outer_product(dx);
//     dd *= ( 1./delgam );
//     MnAlgebraicSymMatrix VggV = Outer_productn(vg);
//     VggV *= ( 1./gvg );
//     Vupd = dd - VggV;
//   }
//   

  double sum_upd = sum_of_elements(Vupd);
  Vupd += V0;
    
//   MnAlgebraicSymMatrix V1 = V0 + Vupd;

  double dcov = 
    0.5*(s0.Error().Dcovar() + sum_upd/sum_of_elements(Vupd));
  
  return MinimumError(Vupd, dcov);
}
*/

  }  // namespace Minuit2

}  // namespace ROOT
