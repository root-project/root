// @(#)root/smatrix:$Id$
// Authors: T. Glebe, L. Moneta    2005

#ifndef ROOT_Math_SMatrixFfwd
#define ROOT_Math_SMatrixFfwd

namespace ROOT { 

namespace Math{ 

   template <class T, unsigned int D1, unsigned int D2, class R> class SMatrix; 

   template <class T, unsigned int D1, unsigned int D2> class MatRepStd; 
   template <class T, unsigned int D> class MatRepSym; 

   typedef SMatrix<float,2,2,MatRepStd<float,2,2> >   SMatrix2F; 
   typedef SMatrix<float,3,3,MatRepStd<float,3,3> >   SMatrix3F; 
   typedef SMatrix<float,4,4,MatRepStd<float,4,4> >   SMatrix4F; 
   typedef SMatrix<float,5,5,MatRepStd<float,5,5> >   SMatrix5F; 
   typedef SMatrix<float,6,6,MatRepStd<float,6,6> >   SMatrix6F; 
   typedef SMatrix<float,7,7,MatRepStd<float,7,7> >   SMatrix7F; 

   typedef SMatrix<float,2,2,MatRepSym<float,2> >     SMatrixSym2F; 
   typedef SMatrix<float,3,3,MatRepSym<float,3> >     SMatrixSym3F; 
   typedef SMatrix<float,4,4,MatRepSym<float,4> >     SMatrixSym4F; 
   typedef SMatrix<float,5,5,MatRepSym<float,5> >     SMatrixSym5F; 
   typedef SMatrix<float,6,6,MatRepSym<float,6> >     SMatrixSym6F; 
   typedef SMatrix<float,7,7,MatRepSym<float,7> >     SMatrixSym7F; 


}  // namespace Math

}  // namespace ROOT

#endif
