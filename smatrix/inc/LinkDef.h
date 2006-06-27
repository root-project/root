// @(#)root/smatrix:$Name:  $:$Id: LinkDef.h,v 1.9 2006/06/02 15:04:54 moneta Exp $
// Authors: L. Moneta    2005  




#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

//#pragma link C++ namespace tvmet;

//#pragma link C++ typedef value_type;

#pragma link C++ namespace ROOT::Math;

#pragma link C++ class ROOT::Math::SMatrixIdentity+;

//generate up to 5x5
#pragma link C++ class ROOT::Math::SMatrix<double,2,2>+;
#pragma link C++ class ROOT::Math::SMatrix<double,3,3>+;
#pragma link C++ class ROOT::Math::SMatrix<double,4,4>+;
#pragma link C++ class ROOT::Math::SMatrix<double,5,5>+;

#pragma link C++ class ROOT::Math::SMatrix<double,4,3>+;
#pragma link C++ class ROOT::Math::SMatrix<double,3,4>+;
#pragma link C++ class ROOT::Math::SMatrix<double,9,7>+;

#pragma link C++ class ROOT::Math::MatRepStd<double,2,2>+;
#pragma link C++ class ROOT::Math::MatRepStd<double,3,3>+;
#pragma link C++ class ROOT::Math::MatRepStd<double,4,4>+;
#pragma link C++ class ROOT::Math::MatRepStd<double,5,5>+;

#pragma link C++ class ROOT::Math::MatRepStd<double,4,3>+;
#pragma link C++ class ROOT::Math::MatRepStd<double,3,4>+;
#pragma link C++ class ROOT::Math::MatRepStd<double,9,7>+;



#pragma link C++ class ROOT::Math::SVector<double,2>+;
#pragma link C++ class ROOT::Math::SVector<double,3>+;
#pragma link C++ class ROOT::Math::SVector<double,4>+;
#pragma link C++ class ROOT::Math::SVector<double,5>+;

#pragma link C++ class ROOT::Math::MatRepSym<double,2>+;
#pragma link C++ class ROOT::Math::MatRepSym<double,3>+;
#pragma link C++ class ROOT::Math::MatRepSym<double,4>+;
#pragma link C++ class ROOT::Math::MatRepSym<double,5>+;

#pragma link C++ struct ROOT::Math::RowOffsets<2>;
#pragma link C++ struct ROOT::Math::RowOffsets<3>;
#pragma link C++ struct ROOT::Math::RowOffsets<4>;
#pragma link C++ struct ROOT::Math::RowOffsets<5>;


#pragma link C++ class ROOT::Math::SMatrix<double,2,2,ROOT::Math::MatRepSym<double,2> >+;
#pragma link C++ class ROOT::Math::SMatrix<double,3,3,ROOT::Math::MatRepSym<double,3> >+;
#pragma link C++ class ROOT::Math::SMatrix<double,4,4,ROOT::Math::MatRepSym<double,4> >+;
#pragma link C++ class ROOT::Math::SMatrix<double,5,5,ROOT::Math::MatRepSym<double,5> >+;
// #pragma link C++ class ROOT::Math::SMatrix<double,3,3>+;
// #pragma link C++ class ROOT::Math::SMatrix<double,4,4>+;
// #pragma link C++ class ROOT::Math::SMatrix<double,5,5>+;

// typedef's 
#pragma link C++ typedef ROOT::Math::SMatrix5D;
#pragma link C++ typedef ROOT::Math::SMatrixSym5D;


#endif
