/// \file
/// \ingroup tutorial_math
/// \notebook -nodraw
/// Example on the usage of the multidimensional integration algorithm of MathMore.
///
/// Please refer to the web documentation for further details: 
///      https://root.cern/manual/math/#numerical-integration
/// To execute the macro type the following:
///
/// ~~~{.cpp}
/// root[0] .x mathmoreIntegrationMultidim.C
/// ~~~
///
/// This tutorial requires having libMathMore built with ROOT.
///
/// To build mathmore you need to have a version of GSL >= 1.8 installed in your system
/// The ROOT configure will automatically find GSL if the script gsl-config (from GSL) is in your PATH,.
/// otherwise you need to configure root with the options --gsl-incdir and --gsl-libdir.
///
/// \macro_image
/// \macro_output
/// \macro_code
///
/// \authors A. Tolosa-Delgado

double f2(const double * x) {
   return x[0] + x[1];
}

int mathmoreIntegrationMultidim() {

   const double RESULT = 1.0;
   const double ERRORLIMIT = 1E-3;
   int status = 0;

   ROOT::Math::Functor wf(&f2,2);
   double a[2] = {0,0};
   double b[2] = {1,1};

   ROOT::Math::IntegratorMultiDim ig(ROOT::Math::IntegrationMultiDim::kADAPTIVE);
   ig.SetFunction(wf);
   double val = ig.Integral(a,b);
   std::cout << "integral result is " << val << std::endl;
   status += std::fabs(val-RESULT) > ERRORLIMIT;

   ROOT::Math::IntegratorMultiDim ig2(ROOT::Math::IntegrationMultiDim::kVEGAS);
   ig2.SetFunction(wf);
   val = ig2.Integral(a,b);
   std::cout << "integral result is " << val << std::endl;
   status += std::fabs(val-RESULT) > ERRORLIMIT;

   ROOT::Math::IntegratorMultiDim ig3(wf,ROOT::Math::IntegrationMultiDim::kPLAIN);
   val = ig3.Integral(a,b);
   std::cout << "integral result is " << val << std::endl;
   status += std::fabs(val-RESULT) > ERRORLIMIT;

   ROOT::Math::IntegratorMultiDim ig4(wf,ROOT::Math::IntegrationMultiDim::kMISER);
   val = ig4.Integral(a,b);
   std::cout << "integral result is " << val << std::endl;
   status += std::fabs(val-RESULT) > ERRORLIMIT;

   return status;
}
