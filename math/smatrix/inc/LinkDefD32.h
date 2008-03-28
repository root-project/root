// @(#)root/smatrix:$Id$
// Authors: L. Moneta    2005  




#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ nestedclass;

//#pragma link C++ namespace tvmet;

//#pragma link C++ typedef value_type;


//#pragma link C++ class ROOT::Math::SMatrixIdentity+;

//generate from 3x3 up to 6x6
#pragma link C++ class ROOT::Math::SMatrix<Double32_t,3,3>+;
#pragma link C++ class ROOT::Math::SMatrix<Double32_t,4,4>+;
#pragma link C++ class ROOT::Math::SMatrix<Double32_t,5,5>+;
#pragma link C++ class ROOT::Math::SMatrix<Double32_t,6,6>+;


#pragma link C++ class ROOT::Math::MatRepStd<Double32_t,3,3>+;
#pragma link C++ class ROOT::Math::MatRepStd<Double32_t,4,4>+;
#pragma link C++ class ROOT::Math::MatRepStd<Double32_t,5,5>+;
#pragma link C++ class ROOT::Math::MatRepStd<Double32_t,6,6>+;


#pragma link C++ class ROOT::Math::SVector<Double32_t,3>+;
#pragma link C++ class ROOT::Math::SVector<Double32_t,4>+;
#pragma link C++ class ROOT::Math::SVector<Double32_t,5>+;
#pragma link C++ class ROOT::Math::SVector<Double32_t,6>+;

#pragma link C++ class ROOT::Math::MatRepSym<Double32_t,3>+;
#pragma link C++ class ROOT::Math::MatRepSym<Double32_t,4>+;
#pragma link C++ class ROOT::Math::MatRepSym<Double32_t,5>+;
#pragma link C++ class ROOT::Math::MatRepSym<Double32_t,6>+;

#pragma link C++ class ROOT::Math::SMatrix<Double32_t,3,3,ROOT::Math::MatRepSym<Double32_t,3> >+;
#pragma link C++ class ROOT::Math::SMatrix<Double32_t,4,4,ROOT::Math::MatRepSym<Double32_t,4> >+;
#pragma link C++ class ROOT::Math::SMatrix<Double32_t,5,5,ROOT::Math::MatRepSym<Double32_t,5> >+;
#pragma link C++ class ROOT::Math::SMatrix<Double32_t,6,6,ROOT::Math::MatRepSym<Double32_t,6> >+;


#endif
