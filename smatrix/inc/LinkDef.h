// @(#)root/smatrix:$Name:  $:$Id: LinkDef.h,v 1.2 2005/12/05 16:33:47 moneta Exp $
// Authors: L. Moneta    2005  




#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

//#pragma link C++ namespace tvmet;

//#pragma link C++ typedef value_type;

#pragma link C++ namespace ROOT::Math;

//generate up to 5x5
#pragma link C++ class ROOT::Math::SMatrix<double,2,2>+;
#pragma link C++ class ROOT::Math::SMatrix<double,3,3>+;
#pragma link C++ class ROOT::Math::SMatrix<double,4,4>+;
#pragma link C++ class ROOT::Math::SMatrix<double,5,5>+;

#pragma link C++ class ROOT::Math::SVector<double,2>+;
#pragma link C++ class ROOT::Math::SVector<double,3>+;
#pragma link C++ class ROOT::Math::SVector<double,4>+;
#pragma link C++ class ROOT::Math::SVector<double,5>+;

#pragma link C++ class ROOT::Math::MatRepSym<double,2>+;
#pragma link C++ class ROOT::Math::MatRepSym<double,3>+;
#pragma link C++ class ROOT::Math::MatRepSym<double,4>+;
#pragma link C++ class ROOT::Math::MatRepSym<double,5>+;

#pragma link C++ class ROOT::Math::SMatrix<double,2,2,ROOT::Math::MatRepSym<double,2> >+;
#pragma link C++ class ROOT::Math::SMatrix<double,3,3,ROOT::Math::MatRepSym<double,3> >+;
#pragma link C++ class ROOT::Math::SMatrix<double,4,4,ROOT::Math::MatRepSym<double,4> >+;
#pragma link C++ class ROOT::Math::SMatrix<double,5,5,ROOT::Math::MatRepSym<double,5> >+;
// #pragma link C++ class ROOT::Math::SMatrix<double,3,3>+;
// #pragma link C++ class ROOT::Math::SMatrix<double,4,4>+;
// #pragma link C++ class ROOT::Math::SMatrix<double,5,5>+;



#endif
