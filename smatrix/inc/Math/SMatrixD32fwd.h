// @(#)root/smatrix:$Name:  $:$Id: SMatrixD32fwd.h,v 1.2 2006/06/29 08:47:51 moneta Exp $
// Authors: T. Glebe, L. Moneta    2005

#ifndef ROOT_Math_SMatrixD32fwd
#define ROOT_Math_SMatrixD32fwd


// redefine Double32_t in case we are not using ROOT 
typedef double Double32_t; 

namespace ROOT { 

namespace Math{ 

   template <class T, unsigned int D1, unsigned int D2, class R> class SMatrix; 

   template <class T, unsigned int D1, unsigned int D2> class MatRepStd; 
   template <class T, unsigned int D> class MatRepSym; 


   typedef SMatrix<Double32_t,2,2,MatRepStd<Double32_t,2,2> > SMatrix2D32; 
   typedef SMatrix<Double32_t,3,3,MatRepStd<Double32_t,3,3> > SMatrix3D32; 
   typedef SMatrix<Double32_t,4,4,MatRepStd<Double32_t,4,4> > SMatrix4D32; 
   typedef SMatrix<Double32_t,5,5,MatRepStd<Double32_t,5,5> > SMatrix5D32; 
   typedef SMatrix<Double32_t,6,6,MatRepStd<Double32_t,6,6> > SMatrix6D32; 
   typedef SMatrix<Double32_t,7,7,MatRepStd<Double32_t,7,7> > SMatrix7D32; 


   typedef SMatrix<Double32_t,2,2,MatRepSym<Double32_t,2> >   SMatrixSym2D32; 
   typedef SMatrix<Double32_t,3,3,MatRepSym<Double32_t,3> >   SMatrixSym3D32; 
   typedef SMatrix<Double32_t,4,4,MatRepSym<Double32_t,4> >   SMatrixSym4D32; 
   typedef SMatrix<Double32_t,5,5,MatRepSym<Double32_t,5> >   SMatrixSym5D32; 
   typedef SMatrix<Double32_t,6,6,MatRepSym<Double32_t,6> >   SMatrixSym6D32; 
   typedef SMatrix<Double32_t,7,7,MatRepSym<Double32_t,7> >   SMatrixSym7D32; 

}  // namespace Math

}  // namespace ROOT

#endif
