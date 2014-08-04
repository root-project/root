// @(#)root/smatrix:$Id$
// Authors: L. Moneta    2005




#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ nestedclass;

//#pragma link C++ namespace tvmet;

//#pragma link C++ typedef value_type;

#pragma link C++ namespace ROOT::Math;

#pragma link C++ class ROOT::Math::SMatrixIdentity+;

//generate up to 7x7

#pragma link C++ class ROOT::Math::SVector<double,2>+;
#pragma link C++ class ROOT::Math::SVector<double,3>+;
#pragma link C++ class ROOT::Math::SVector<double,4>+;
#pragma link C++ class ROOT::Math::SVector<double,5>+;
#pragma link C++ class ROOT::Math::SVector<double,6>+;
#pragma link C++ class ROOT::Math::SVector<double,7>+;
// #pragma link C++ class ROOT::Math::SVector<double,8>+;
// #pragma link C++ class ROOT::Math::SVector<double,9>+;
// #pragma link C++ class ROOT::Math::SVector<double,10>+;


#pragma link C++ class ROOT::Math::MatRepStd<double,2,2>+;
#pragma link C++ class ROOT::Math::MatRepStd<double,3,3>+;
#pragma link C++ class ROOT::Math::MatRepStd<double,4,4>+;
#pragma link C++ class ROOT::Math::MatRepStd<double,5,5>+;
#pragma link C++ class ROOT::Math::MatRepStd<double,6,6>+;
#pragma link C++ class ROOT::Math::MatRepStd<double,7,7>+;
// #pragma link C++ class ROOT::Math::MatRepStd<double,8,8>+;
// #pragma link C++ class ROOT::Math::MatRepStd<double,9,9>+;
// #pragma link C++ class ROOT::Math::MatRepStd<double,10,10>+;




#pragma link C++ class ROOT::Math::MatRepStd<double,4,3>+;
#pragma link C++ class ROOT::Math::MatRepStd<double,3,4>+;
#pragma link C++ class ROOT::Math::MatRepStd<double,9,7>+;

#pragma link C++ class ROOT::Math::SMatrix<double,2,2>+;
#pragma link C++ class ROOT::Math::SMatrix<double,3,3>+;
#pragma link C++ class ROOT::Math::SMatrix<double,4,4>+;
#pragma link C++ class ROOT::Math::SMatrix<double,5,5>+;
#pragma link C++ class ROOT::Math::SMatrix<double,6,6>+;
#pragma link C++ class ROOT::Math::SMatrix<double,7,7>+;
// #pragma link C++ class ROOT::Math::SMatrix<double,8,8>+;
// #pragma link C++ class ROOT::Math::SMatrix<double,9,9>+;
// #pragma link C++ class ROOT::Math::SMatrix<double,10,10>+;

#pragma link C++ class ROOT::Math::SMatrix<double,2,2>::SMatrixRow;
#pragma link C++ class ROOT::Math::SMatrix<double,3,3>::SMatrixRow;
#pragma link C++ class ROOT::Math::SMatrix<double,4,4>::SMatrixRow;
#pragma link C++ class ROOT::Math::SMatrix<double,5,5>::SMatrixRow;
#pragma link C++ class ROOT::Math::SMatrix<double,6,6>::SMatrixRow;
#pragma link C++ class ROOT::Math::SMatrix<double,7,7>::SMatrixRow;

#pragma link C++ class ROOT::Math::SMatrix<double,2,2>::SMatrixRow_const;
#pragma link C++ class ROOT::Math::SMatrix<double,3,3>::SMatrixRow_const;
#pragma link C++ class ROOT::Math::SMatrix<double,4,4>::SMatrixRow_const;
#pragma link C++ class ROOT::Math::SMatrix<double,5,5>::SMatrixRow_const;
#pragma link C++ class ROOT::Math::SMatrix<double,6,6>::SMatrixRow_const;
#pragma link C++ class ROOT::Math::SMatrix<double,7,7>::SMatrixRow_const;


#pragma link C++ class ROOT::Math::SMatrix<double,4,3>+;
#pragma link C++ class ROOT::Math::SMatrix<double,3,4>+;
#pragma link C++ class ROOT::Math::SMatrix<double,9,7>+;

#pragma link C++ class ROOT::Math::SMatrix<double,4,3>::SMatrixRow;
#pragma link C++ class ROOT::Math::SMatrix<double,3,4>::SMatrixRow;
#pragma link C++ class ROOT::Math::SMatrix<double,9,7>::SMatrixRow;

#pragma link C++ class ROOT::Math::SMatrix<double,4,3>::SMatrixRow_const;
#pragma link C++ class ROOT::Math::SMatrix<double,3,4>::SMatrixRow_const;
#pragma link C++ class ROOT::Math::SMatrix<double,9,7>::SMatrixRow_const;


#pragma link C++ class ROOT::Math::MatRepSym<double,2>+;
#pragma link C++ class ROOT::Math::MatRepSym<double,3>+;
#pragma link C++ class ROOT::Math::MatRepSym<double,4>+;
#pragma link C++ class ROOT::Math::MatRepSym<double,5>+;
#pragma link C++ class ROOT::Math::MatRepSym<double,6>+;
#pragma link C++ class ROOT::Math::MatRepSym<double,7>+;
// #pragma link C++ class ROOT::Math::MatRepSym<double,8>+;
// #pragma link C++ class ROOT::Math::MatRepSym<double,9>+;
// #pragma link C++ class ROOT::Math::MatRepSym<double,10>+;

#pragma link C++ struct ROOT::Math::RowOffsets<2>;
#pragma link C++ struct ROOT::Math::RowOffsets<3>;
#pragma link C++ struct ROOT::Math::RowOffsets<4>;
#pragma link C++ struct ROOT::Math::RowOffsets<5>;
#pragma link C++ struct ROOT::Math::RowOffsets<6>;
#pragma link C++ struct ROOT::Math::RowOffsets<7>;
// #pragma link C++ struct ROOT::Math::RowOffsets<8>;
// #pragma link C++ struct ROOT::Math::RowOffsets<9>;
// #pragma link C++ struct ROOT::Math::RowOffsets<10>;


#pragma link C++ class ROOT::Math::SMatrix<double,2,2,ROOT::Math::MatRepSym<double,2> >+;
#pragma link C++ class ROOT::Math::SMatrix<double,3,3,ROOT::Math::MatRepSym<double,3> >+;
#pragma link C++ class ROOT::Math::SMatrix<double,4,4,ROOT::Math::MatRepSym<double,4> >+;
#pragma link C++ class ROOT::Math::SMatrix<double,5,5,ROOT::Math::MatRepSym<double,5> >+;
#pragma link C++ class ROOT::Math::SMatrix<double,6,6,ROOT::Math::MatRepSym<double,6> >+;
#pragma link C++ class ROOT::Math::SMatrix<double,7,7,ROOT::Math::MatRepSym<double,7> >+;


#pragma link C++ class ROOT::Math::SMatrix<double,2,2,ROOT::Math::MatRepSym<double,2> >::SMatrixRow;
#pragma link C++ class ROOT::Math::SMatrix<double,3,3,ROOT::Math::MatRepSym<double,3> >::SMatrixRow;
#pragma link C++ class ROOT::Math::SMatrix<double,4,4,ROOT::Math::MatRepSym<double,4> >::SMatrixRow;
#pragma link C++ class ROOT::Math::SMatrix<double,5,5,ROOT::Math::MatRepSym<double,5> >::SMatrixRow;
#pragma link C++ class ROOT::Math::SMatrix<double,6,6,ROOT::Math::MatRepSym<double,6> >::SMatrixRow;
#pragma link C++ class ROOT::Math::SMatrix<double,7,7,ROOT::Math::MatRepSym<double,7> >::SMatrixRow;

#pragma link C++ class ROOT::Math::SMatrix<double,2,2,ROOT::Math::MatRepSym<double,2> >::SMatrixRow_const;
#pragma link C++ class ROOT::Math::SMatrix<double,3,3,ROOT::Math::MatRepSym<double,3> >::SMatrixRow_const;
#pragma link C++ class ROOT::Math::SMatrix<double,4,4,ROOT::Math::MatRepSym<double,4> >::SMatrixRow_const;
#pragma link C++ class ROOT::Math::SMatrix<double,5,5,ROOT::Math::MatRepSym<double,5> >::SMatrixRow_const;
#pragma link C++ class ROOT::Math::SMatrix<double,6,6,ROOT::Math::MatRepSym<double,6> >::SMatrixRow_const;
#pragma link C++ class ROOT::Math::SMatrix<double,7,7,ROOT::Math::MatRepSym<double,7> >::SMatrixRow_const;


// #pragma link C++ class ROOT::Math::SMatrix<double,8,8,ROOT::Math::MatRepSym<double,8> >+;
// #pragma link C++ class ROOT::Math::SMatrix<double,9,9,ROOT::Math::MatRepSym<double,9> >+;
// #pragma link C++ class ROOT::Math::SMatrix<double,10,10,ROOT::Math::MatRepSym<double,10> >+;

// typedef's for  double matrices
#pragma link C++ typedef ROOT::Math::SMatrix2D;
#pragma link C++ typedef ROOT::Math::SMatrix3D;
#pragma link C++ typedef ROOT::Math::SMatrix4D;
#pragma link C++ typedef ROOT::Math::SMatrix5D;
#pragma link C++ typedef ROOT::Math::SMatrix6D;
#pragma link C++ typedef ROOT::Math::SMatrix7D;

#pragma link C++ typedef ROOT::Math::SMatrixSym2D;
#pragma link C++ typedef ROOT::Math::SMatrixSym3D;
#pragma link C++ typedef ROOT::Math::SMatrixSym4D;
#pragma link C++ typedef ROOT::Math::SMatrixSym5D;
#pragma link C++ typedef ROOT::Math::SMatrixSym6D;
#pragma link C++ typedef ROOT::Math::SMatrixSym7D;



//now for float

#pragma link C++ class ROOT::Math::SVector<float,2>+;
#pragma link C++ class ROOT::Math::SVector<float,3>+;
#pragma link C++ class ROOT::Math::SVector<float,4>+;
#pragma link C++ class ROOT::Math::SVector<float,5>+;
#pragma link C++ class ROOT::Math::SVector<float,6>+;
#pragma link C++ class ROOT::Math::SVector<float,7>+;
// #pragma link C++ class ROOT::Math::SVector<float,8>+;
// #pragma link C++ class ROOT::Math::SVector<float,9>+;
// #pragma link C++ class ROOT::Math::SVector<float,10>+;


#pragma link C++ class ROOT::Math::MatRepStd<float,2,2>+;
#pragma link C++ class ROOT::Math::MatRepStd<float,3,3>+;
#pragma link C++ class ROOT::Math::MatRepStd<float,4,4>+;
#pragma link C++ class ROOT::Math::MatRepStd<float,5,5>+;
#pragma link C++ class ROOT::Math::MatRepStd<float,6,6>+;
#pragma link C++ class ROOT::Math::MatRepStd<float,7,7>+;
// #pragma link C++ class ROOT::Math::MatRepStd<float,8,8>+;
// #pragma link C++ class ROOT::Math::MatRepStd<float,9,9>+;
// #pragma link C++ class ROOT::Math::MatRepStd<float,10,10>+;

#pragma link C++ class ROOT::Math::MatRepStd<float,4,3>+;
#pragma link C++ class ROOT::Math::MatRepStd<float,3,4>+;
#pragma link C++ class ROOT::Math::MatRepStd<float,9,7>+;

#pragma link C++ class ROOT::Math::SMatrix<float,2,2>+;
#pragma link C++ class ROOT::Math::SMatrix<float,3,3>+;
#pragma link C++ class ROOT::Math::SMatrix<float,4,4>+;
#pragma link C++ class ROOT::Math::SMatrix<float,5,5>+;
#pragma link C++ class ROOT::Math::SMatrix<float,6,6>+;
#pragma link C++ class ROOT::Math::SMatrix<float,7,7>+;
// #pragma link C++ class ROOT::Math::SMatrix<float,8,8>+;
// #pragma link C++ class ROOT::Math::SMatrix<float,9,9>+;
// #pragma link C++ class ROOT::Math::SMatrix<float,10,10>+;

#pragma link C++ class ROOT::Math::SMatrix<float,4,3>+;
#pragma link C++ class ROOT::Math::SMatrix<float,3,4>+;
#pragma link C++ class ROOT::Math::SMatrix<float,9,7>+;

#pragma link C++ class ROOT::Math::SMatrix<float,2,2>::SMatrixRow;
#pragma link C++ class ROOT::Math::SMatrix<float,3,3>::SMatrixRow;
#pragma link C++ class ROOT::Math::SMatrix<float,4,4>::SMatrixRow;
#pragma link C++ class ROOT::Math::SMatrix<float,5,5>::SMatrixRow;
#pragma link C++ class ROOT::Math::SMatrix<float,6,6>::SMatrixRow;
#pragma link C++ class ROOT::Math::SMatrix<float,7,7>::SMatrixRow;

#pragma link C++ class ROOT::Math::SMatrix<float,2,2>::SMatrixRow_const;
#pragma link C++ class ROOT::Math::SMatrix<float,3,3>::SMatrixRow_const;
#pragma link C++ class ROOT::Math::SMatrix<float,4,4>::SMatrixRow_const;
#pragma link C++ class ROOT::Math::SMatrix<float,5,5>::SMatrixRow_const;
#pragma link C++ class ROOT::Math::SMatrix<float,6,6>::SMatrixRow_const;
#pragma link C++ class ROOT::Math::SMatrix<float,7,7>::SMatrixRow_const;

#pragma link C++ class ROOT::Math::SMatrix<float,4,3>::SMatrixRow;
#pragma link C++ class ROOT::Math::SMatrix<float,3,4>::SMatrixRow;
#pragma link C++ class ROOT::Math::SMatrix<float,9,7>::SMatrixRow;

#pragma link C++ class ROOT::Math::SMatrix<float,4,3>::SMatrixRow_const;
#pragma link C++ class ROOT::Math::SMatrix<float,3,4>::SMatrixRow_const;
#pragma link C++ class ROOT::Math::SMatrix<float,9,7>::SMatrixRow_const;


#pragma link C++ class ROOT::Math::MatRepSym<float,2>+;
#pragma link C++ class ROOT::Math::MatRepSym<float,3>+;
#pragma link C++ class ROOT::Math::MatRepSym<float,4>+;
#pragma link C++ class ROOT::Math::MatRepSym<float,5>+;
#pragma link C++ class ROOT::Math::MatRepSym<float,6>+;
#pragma link C++ class ROOT::Math::MatRepSym<float,7>+;
// #pragma link C++ class ROOT::Math::MatRepSym<float,8>+;
// #pragma link C++ class ROOT::Math::MatRepSym<float,9>+;
// #pragma link C++ class ROOT::Math::MatRepSym<float,10>+;

#pragma link C++ struct ROOT::Math::RowOffsets<2>;
#pragma link C++ struct ROOT::Math::RowOffsets<3>;
#pragma link C++ struct ROOT::Math::RowOffsets<4>;
#pragma link C++ struct ROOT::Math::RowOffsets<5>;
#pragma link C++ struct ROOT::Math::RowOffsets<6>;
#pragma link C++ struct ROOT::Math::RowOffsets<7>;
// #pragma link C++ struct ROOT::Math::RowOffsets<8>;
// #pragma link C++ struct ROOT::Math::RowOffsets<9>;
// #pragma link C++ struct ROOT::Math::RowOffsets<10>;


#pragma link C++ class ROOT::Math::SMatrix<float,2,2,ROOT::Math::MatRepSym<float,2> >+;
#pragma link C++ class ROOT::Math::SMatrix<float,3,3,ROOT::Math::MatRepSym<float,3> >+;
#pragma link C++ class ROOT::Math::SMatrix<float,4,4,ROOT::Math::MatRepSym<float,4> >+;
#pragma link C++ class ROOT::Math::SMatrix<float,5,5,ROOT::Math::MatRepSym<float,5> >+;
#pragma link C++ class ROOT::Math::SMatrix<float,6,6,ROOT::Math::MatRepSym<float,6> >+;
#pragma link C++ class ROOT::Math::SMatrix<float,7,7,ROOT::Math::MatRepSym<float,7> >+;
// #pragma link C++ class ROOT::Math::SMatrix<float,8,8,ROOT::Math::MatRepSym<float,8> >+;
// #pragma link C++ class ROOT::Math::SMatrix<float,9,9,ROOT::Math::MatRepSym<float,9> >+;
// #pragma link C++ class ROOT::Math::SMatrix<float,10,10,ROOT::Math::MatRepSym<float,10> >+;


#pragma link C++ class ROOT::Math::SMatrix<float,2,2,ROOT::Math::MatRepSym<float,2> >::SMatrixRow;
#pragma link C++ class ROOT::Math::SMatrix<float,3,3,ROOT::Math::MatRepSym<float,3> >::SMatrixRow;
#pragma link C++ class ROOT::Math::SMatrix<float,4,4,ROOT::Math::MatRepSym<float,4> >::SMatrixRow;
#pragma link C++ class ROOT::Math::SMatrix<float,5,5,ROOT::Math::MatRepSym<float,5> >::SMatrixRow;
#pragma link C++ class ROOT::Math::SMatrix<float,6,6,ROOT::Math::MatRepSym<float,6> >::SMatrixRow;
#pragma link C++ class ROOT::Math::SMatrix<float,7,7,ROOT::Math::MatRepSym<float,7> >::SMatrixRow;

#pragma link C++ class ROOT::Math::SMatrix<float,2,2,ROOT::Math::MatRepSym<float,2> >::SMatrixRow_const;
#pragma link C++ class ROOT::Math::SMatrix<float,3,3,ROOT::Math::MatRepSym<float,3> >::SMatrixRow_const;
#pragma link C++ class ROOT::Math::SMatrix<float,4,4,ROOT::Math::MatRepSym<float,4> >::SMatrixRow_const;
#pragma link C++ class ROOT::Math::SMatrix<float,5,5,ROOT::Math::MatRepSym<float,5> >::SMatrixRow_const;
#pragma link C++ class ROOT::Math::SMatrix<float,6,6,ROOT::Math::MatRepSym<float,6> >::SMatrixRow_const;
#pragma link C++ class ROOT::Math::SMatrix<float,7,7,ROOT::Math::MatRepSym<float,7> >::SMatrixRow_const;


// typedef's for  float matrices
#pragma link C++ typedef ROOT::Math::SMatrix2F;
#pragma link C++ typedef ROOT::Math::SMatrix3F;
#pragma link C++ typedef ROOT::Math::SMatrix4F;
#pragma link C++ typedef ROOT::Math::SMatrix5F;
#pragma link C++ typedef ROOT::Math::SMatrix6F;
#pragma link C++ typedef ROOT::Math::SMatrix7F;

#pragma link C++ typedef ROOT::Math::SMatrixSym2F;
#pragma link C++ typedef ROOT::Math::SMatrixSym3F;
#pragma link C++ typedef ROOT::Math::SMatrixSym4F;
#pragma link C++ typedef ROOT::Math::SMatrixSym5F;
#pragma link C++ typedef ROOT::Math::SMatrixSym6F;
#pragma link C++ typedef ROOT::Math::SMatrixSym7F;


#endif
