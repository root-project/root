// @(#)root/smatrix:$Name:  $:$Id: LinkDef.h,v 1.9 2006/06/02 15:04:54 moneta Exp $
// Authors: L. Moneta    2005  




#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

//#pragma link C++ namespace tvmet;

//#pragma link C++ typedef value_type;


//#pragma link C++ class ROOT::Math::SMatrixIdentity+;

//generate up to 5x5
#pragma link C++ class ROOT::Math::SMatrix<Double32_t,2,2>+;
#pragma link C++ class ROOT::Math::SMatrix<Double32_t,3,3>+;
#pragma link C++ class ROOT::Math::SMatrix<Double32_t,4,4>+;
#pragma link C++ class ROOT::Math::SMatrix<Double32_t,5,5>+;

#pragma link C++ class ROOT::Math::SMatrix<Double32_t,4,3>+;
#pragma link C++ class ROOT::Math::SMatrix<Double32_t,3,4>+;
#pragma link C++ class ROOT::Math::SMatrix<Double32_t,9,7>+;

#pragma link C++ class ROOT::Math::MatRepStd<Double32_t,2,2>+;
#pragma link C++ class ROOT::Math::MatRepStd<Double32_t,3,3>+;
#pragma link C++ class ROOT::Math::MatRepStd<Double32_t,4,4>+;
#pragma link C++ class ROOT::Math::MatRepStd<Double32_t,5,5>+;

#pragma link C++ class ROOT::Math::MatRepStd<Double32_t,4,3>+;
#pragma link C++ class ROOT::Math::MatRepStd<Double32_t,3,4>+;
#pragma link C++ class ROOT::Math::MatRepStd<Double32_t,9,7>+;



#pragma link C++ class ROOT::Math::SVector<Double32_t,2>+;
#pragma link C++ class ROOT::Math::SVector<Double32_t,3>+;
#pragma link C++ class ROOT::Math::SVector<Double32_t,4>+;
#pragma link C++ class ROOT::Math::SVector<Double32_t,5>+;

#pragma link C++ class ROOT::Math::MatRepSym<Double32_t,2>+;
#pragma link C++ class ROOT::Math::MatRepSym<Double32_t,3>+;
#pragma link C++ class ROOT::Math::MatRepSym<Double32_t,4>+;
#pragma link C++ class ROOT::Math::MatRepSym<Double32_t,5>+;


#pragma link C++ class ROOT::Math::SMatrix<Double32_t,2,2,ROOT::Math::MatRepSym<Double32_t,2> >+;
#pragma link C++ class ROOT::Math::SMatrix<Double32_t,3,3,ROOT::Math::MatRepSym<Double32_t,3> >+;
#pragma link C++ class ROOT::Math::SMatrix<Double32_t,4,4,ROOT::Math::MatRepSym<Double32_t,4> >+;
#pragma link C++ class ROOT::Math::SMatrix<Double32_t,5,5,ROOT::Math::MatRepSym<Double32_t,5> >+;

#pragma link C++ typedef ROOT::Math::SMatrix5D32;
#pragma link C++ typedef ROOT::Math::SMatrixSym5D32;


#endif
