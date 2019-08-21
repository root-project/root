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
#pragma link C++ class    ROOT::Math::SMatrix<Double32_t,3,3>+;
#pragma read sourceClass="ROOT::Math::SMatrix<double,3,3>"    \
             targetClass="ROOT::Math::SMatrix<Double32_t,3,3>";
#pragma read sourceClass="ROOT::Math::SMatrix<float,3,3>"     \
             targetClass="ROOT::Math::SMatrix<Double32_t,3,3>";
#pragma read sourceClass="ROOT::Math::SMatrix<Float16_t,3,3>" \
             targetClass="ROOT::Math::SMatrix<Double32_t,3,3>";

#pragma link C++ class    ROOT::Math::SMatrix<Double32_t,4,4>+;
#pragma read sourceClass="ROOT::Math::SMatrix<double,4,4>"    \
             targetClass="ROOT::Math::SMatrix<Double32_t,4,4>";
#pragma read sourceClass="ROOT::Math::SMatrix<float,4,4>"     \
             targetClass="ROOT::Math::SMatrix<Double32_t,4,4>";
#pragma read sourceClass="ROOT::Math::SMatrix<Float16_t,4,4>" \
             targetClass="ROOT::Math::SMatrix<Double32_t,4,4>";

#pragma link C++ class    ROOT::Math::SMatrix<Double32_t,5,5>+;
#pragma read sourceClass="ROOT::Math::SMatrix<double,5,5>"    \
             targetClass="ROOT::Math::SMatrix<Double32_t,5,5>";
#pragma read sourceClass="ROOT::Math::SMatrix<float,5,5>"     \
             targetClass="ROOT::Math::SMatrix<Double32_t,5,5>";
#pragma read sourceClass="ROOT::Math::SMatrix<Float16_t,5,5>" \
             targetClass="ROOT::Math::SMatrix<Double32_t,5,5>";

#pragma link C++ class    ROOT::Math::SMatrix<Double32_t,6,6>+;
#pragma read sourceClass="ROOT::Math::SMatrix<double,6,6>"    \
             targetClass="ROOT::Math::SMatrix<Double32_t,6,6>";
#pragma read sourceClass="ROOT::Math::SMatrix<float,6,6>"     \
             targetClass="ROOT::Math::SMatrix<Double32_t,6,6>";
#pragma read sourceClass="ROOT::Math::SMatrix<Float16_t,6,6>" \
             targetClass="ROOT::Math::SMatrix<Double32_t,6,6>";



#pragma link C++ class    ROOT::Math::MatRepStd<Double32_t,3,3>+;
#pragma read sourceClass="ROOT::Math::MatRepStd<double,3,3>"    \
             targetClass="ROOT::Math::MatRepStd<Double32_t,3,3>";
#pragma read sourceClass="ROOT::Math::MatRepStd<float,3,3>"     \
             targetClass="ROOT::Math::MatRepStd<Double32_t,3,3>";
#pragma read sourceClass="ROOT::Math::MatRepStd<Float16_t,3,3>" \
             targetClass="ROOT::Math::MatRepStd<Double32_t,3,3>";

#pragma link C++ class    ROOT::Math::MatRepStd<Double32_t,4,4>+;
#pragma read sourceClass="ROOT::Math::MatRepStd<double,4,4>"    \
             targetClass="ROOT::Math::MatRepStd<Double32_t,4,4>";
#pragma read sourceClass="ROOT::Math::MatRepStd<float,4,4>"     \
             targetClass="ROOT::Math::MatRepStd<Double32_t,4,4>";
#pragma read sourceClass="ROOT::Math::MatRepStd<Float16_t,4,4>" \
             targetClass="ROOT::Math::MatRepStd<Double32_t,4,4>";

#pragma link C++ class    ROOT::Math::MatRepStd<Double32_t,5,5>+;
#pragma read sourceClass="ROOT::Math::MatRepStd<double,5,5>"    \
             targetClass="ROOT::Math::MatRepStd<Double32_t,5,5>";
#pragma read sourceClass="ROOT::Math::MatRepStd<float,5,5>"     \
             targetClass="ROOT::Math::MatRepStd<Double32_t,5,5>";
#pragma read sourceClass="ROOT::Math::MatRepStd<Float16_t,5,5>" \
             targetClass="ROOT::Math::MatRepStd<Double32_t,5,5>";

#pragma link C++ class    ROOT::Math::MatRepStd<Double32_t,6,6>+;
#pragma read sourceClass="ROOT::Math::MatRepStd<double,6,6>"    \
             targetClass="ROOT::Math::MatRepStd<Double32_t,6,6>";
#pragma read sourceClass="ROOT::Math::MatRepStd<float,6,6>"     \
             targetClass="ROOT::Math::MatRepStd<Double32_t,6,6>";
#pragma read sourceClass="ROOT::Math::MatRepStd<Float16_t,6,6>" \
             targetClass="ROOT::Math::MatRepStd<Double32_t,6,6>";



#pragma link C++ class    ROOT::Math::SVector<Double32_t,3>+;
#pragma read sourceClass="ROOT::Math::SVector<double,3>"    \
             targetClass="ROOT::Math::SVector<Double32_t,3>";
#pragma read sourceClass="ROOT::Math::SVector<float,3>"     \
             targetClass="ROOT::Math::SVector<Double32_t,3>";
#pragma read sourceClass="ROOT::Math::SVector<Float16_t,3>" \
             targetClass="ROOT::Math::SVector<Double32_t,3>";

#pragma link C++ class    ROOT::Math::SVector<Double32_t,4>+;
#pragma read sourceClass="ROOT::Math::SVector<double,4>"    \
             targetClass="ROOT::Math::SVector<Double32_t,4>";
#pragma read sourceClass="ROOT::Math::SVector<float,4>"     \
             targetClass="ROOT::Math::SVector<Double32_t,4>";
#pragma read sourceClass="ROOT::Math::SVector<Float16_t,4>" \
             targetClass="ROOT::Math::SVector<Double32_t,4>";

#pragma link C++ class    ROOT::Math::SVector<Double32_t,5>+;
#pragma read sourceClass="ROOT::Math::SVector<double,5>"    \
             targetClass="ROOT::Math::SVector<Double32_t,5>";
#pragma read sourceClass="ROOT::Math::SVector<float,5>"     \
             targetClass="ROOT::Math::SVector<Double32_t,5>";
#pragma read sourceClass="ROOT::Math::SVector<Float16_t,5>" \
             targetClass="ROOT::Math::SVector<Double32_t,5>";

#pragma link C++ class    ROOT::Math::SVector<Double32_t,6>+;
#pragma read sourceClass="ROOT::Math::SVector<double,6>"    \
             targetClass="ROOT::Math::SVector<Double32_t,6>";
#pragma read sourceClass="ROOT::Math::SVector<float,6>"     \
             targetClass="ROOT::Math::SVector<Double32_t,6>";
#pragma read sourceClass="ROOT::Math::SVector<Float16_t,6>" \
             targetClass="ROOT::Math::SVector<Double32_t,6>";


#pragma link C++ class    ROOT::Math::MatRepSym<Double32_t,3>+;
#pragma read sourceClass="ROOT::Math::MatRepSym<double,3>"    \
             targetClass="ROOT::Math::MatRepSym<Double32_t,3>";
#pragma read sourceClass="ROOT::Math::MatRepSym<float,3>"     \
             targetClass="ROOT::Math::MatRepSym<Double32_t,3>";
#pragma read sourceClass="ROOT::Math::MatRepSym<Float16_t,3>" \
             targetClass="ROOT::Math::MatRepSym<Double32_t,3>";

#pragma link C++ class    ROOT::Math::MatRepSym<Double32_t,4>+;
#pragma read sourceClass="ROOT::Math::MatRepSym<double,4>"    \
             targetClass="ROOT::Math::MatRepSym<Double32_t,4>";
#pragma read sourceClass="ROOT::Math::MatRepSym<float,4>"     \
             targetClass="ROOT::Math::MatRepSym<Double32_t,4>";
#pragma read sourceClass="ROOT::Math::MatRepSym<Float16_t,4>" \
             targetClass="ROOT::Math::MatRepSym<Double32_t,4>";

#pragma link C++ class    ROOT::Math::MatRepSym<Double32_t,5>+;
#pragma read sourceClass="ROOT::Math::MatRepSym<double,5>"    \
             targetClass="ROOT::Math::MatRepSym<Double32_t,5>";
#pragma read sourceClass="ROOT::Math::MatRepSym<float,5>"     \
             targetClass="ROOT::Math::MatRepSym<Double32_t,5>";
#pragma read sourceClass="ROOT::Math::MatRepSym<Float16_t,5>" \
             targetClass="ROOT::Math::MatRepSym<Double32_t,5>";

#pragma link C++ class    ROOT::Math::MatRepSym<Double32_t,6>+;
#pragma read sourceClass="ROOT::Math::MatRepSym<double,6>"    \
             targetClass="ROOT::Math::MatRepSym<Double32_t,6>";
#pragma read sourceClass="ROOT::Math::MatRepSym<float,6>"     \
             targetClass="ROOT::Math::MatRepSym<Double32_t,6>";
#pragma read sourceClass="ROOT::Math::MatRepSym<Float16_t,6>" \
             targetClass="ROOT::Math::MatRepSym<Double32_t,6>";


#pragma link C++ class    ROOT::Math::SMatrix<Double32_t,3,3,ROOT::Math::MatRepSym<Double32_t,3> >+;
#pragma read sourceClass="ROOT::Math::SMatrix<Double32_t,3,3,ROOT::Math::MatRepSym<double,3> >"    \
             targetClass="ROOT::Math::SMatrix<Double32_t,3,3,ROOT::Math::MatRepSym<Double32_t,3> >";
#pragma read sourceClass="ROOT::Math::SMatrix<Double32_t,3,3,ROOT::Math::MatRepSym<float,3> >"     \
             targetClass="ROOT::Math::SMatrix<Double32_t,3,3,ROOT::Math::MatRepSym<Double32_t,3> >";
#pragma read sourceClass="ROOT::Math::SMatrix<Double32_t,3,3,ROOT::Math::MatRepSym<Float16_t,3> >" \
             targetClass="ROOT::Math::SMatrix<Double32_t,3,3,ROOT::Math::MatRepSym<Double32_t,3> >";

#pragma link C++ class    ROOT::Math::SMatrix<Double32_t,4,4,ROOT::Math::MatRepSym<Double32_t,4> >+;
#pragma read sourceClass="ROOT::Math::SMatrix<Double32_t,4,4,ROOT::Math::MatRepSym<double,4> >"    \
             targetClass="ROOT::Math::SMatrix<Double32_t,4,4,ROOT::Math::MatRepSym<Double32_t,4> >";
#pragma read sourceClass="ROOT::Math::SMatrix<Double32_t,4,4,ROOT::Math::MatRepSym<float,4> >"     \
             targetClass="ROOT::Math::SMatrix<Double32_t,4,4,ROOT::Math::MatRepSym<Double32_t,4> >";
#pragma read sourceClass="ROOT::Math::SMatrix<Double32_t,4,4,ROOT::Math::MatRepSym<Float16_t,4> >" \
             targetClass="ROOT::Math::SMatrix<Double32_t,4,4,ROOT::Math::MatRepSym<Double32_t,4> >";

#pragma link C++ class    ROOT::Math::SMatrix<Double32_t,5,5,ROOT::Math::MatRepSym<Double32_t,5> >+;
#pragma read sourceClass="ROOT::Math::SMatrix<Double32_t,5,5,ROOT::Math::MatRepSym<double,5> >"    \
             targetClass="ROOT::Math::SMatrix<Double32_t,5,5,ROOT::Math::MatRepSym<Double32_t,5> >";
#pragma read sourceClass="ROOT::Math::SMatrix<Double32_t,5,5,ROOT::Math::MatRepSym<float,5> >"     \
             targetClass="ROOT::Math::SMatrix<Double32_t,5,5,ROOT::Math::MatRepSym<Double32_t,5> >";
#pragma read sourceClass="ROOT::Math::SMatrix<Double32_t,5,5,ROOT::Math::MatRepSym<Float16_t,5> >" \
             targetClass="ROOT::Math::SMatrix<Double32_t,5,5,ROOT::Math::MatRepSym<Double32_t,5> >";

#pragma link C++ class    ROOT::Math::SMatrix<Double32_t,6,6,ROOT::Math::MatRepSym<Double32_t,6> >+;
#pragma read sourceClass="ROOT::Math::SMatrix<Double32_t,6,6,ROOT::Math::MatRepSym<double,6> >"    \
             targetClass="ROOT::Math::SMatrix<Double32_t,6,6,ROOT::Math::MatRepSym<Double32_t,6> >";
#pragma read sourceClass="ROOT::Math::SMatrix<Double32_t,6,6,ROOT::Math::MatRepSym<float,6> >"     \
             targetClass="ROOT::Math::SMatrix<Double32_t,6,6,ROOT::Math::MatRepSym<Double32_t,6> >";
#pragma read sourceClass="ROOT::Math::SMatrix<Double32_t,6,6,ROOT::Math::MatRepSym<Float16_t,6> >" \
             targetClass="ROOT::Math::SMatrix<Double32_t,6,6,ROOT::Math::MatRepSym<Double32_t,6> >";



#endif
