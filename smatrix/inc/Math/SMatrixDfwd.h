// @(#)root/smatrix:$Id$
// Authors: T. Glebe, L. Moneta    2005

#ifndef ROOT_Math_SMatrixDfwd
#define ROOT_Math_SMatrixDfwd

namespace ROOT { 

namespace Math{ 

   template <class T, unsigned int D1, unsigned int D2, class R> class SMatrix; 

   template <class T, unsigned int D1, unsigned int D2> class MatRepStd; 
   template <class T, unsigned int D> class MatRepSym; 

   typedef SMatrix<double,2,2,MatRepStd<double,2,2> > SMatrix2D; 
   typedef SMatrix<double,3,3,MatRepStd<double,3,3> > SMatrix3D; 
   typedef SMatrix<double,4,4,MatRepStd<double,4,4> > SMatrix4D; 
   typedef SMatrix<double,5,5,MatRepStd<double,5,5> > SMatrix5D; 
   typedef SMatrix<double,6,6,MatRepStd<double,6,6> > SMatrix6D; 
   typedef SMatrix<double,7,7,MatRepStd<double,7,7> > SMatrix7D; 


   typedef SMatrix<double,2,2,MatRepSym<double,2> >   SMatrixSym2D; 
   typedef SMatrix<double,3,3,MatRepSym<double,3> >   SMatrixSym3D; 
   typedef SMatrix<double,4,4,MatRepSym<double,4> >   SMatrixSym4D; 
   typedef SMatrix<double,5,5,MatRepSym<double,5> >   SMatrixSym5D; 
   typedef SMatrix<double,6,6,MatRepSym<double,6> >   SMatrixSym6D; 
   typedef SMatrix<double,7,7,MatRepSym<double,7> >   SMatrixSym7D; 

}  // namespace Math

}  // namespace ROOT


#endif
