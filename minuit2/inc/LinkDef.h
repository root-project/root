// @(#)root/minuit2:$Name:  $:$Id: LinkDef.h,v 1.2 2005/12/01 10:26:05 moneta Exp $
// Author: L. Moneta    10/2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 ROOT Foundation,  CERN/PH-SFT                   *
 *                                                                    *
 **********************************************************************/


#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

//#pragma link C++ global gMinuit;
#pragma link C++ global gMinuit2;
#pragma link C++ global gFumili2;

#pragma link C++ class TFitterMinuit;
#pragma link C++ class TFitterFumili;
#pragma link C++ class TFcnAdapter;

//#pragma link C++ namespace ROOT::Minuit2;

#pragma link C++ class ROOT::Minuit2::FCNBase;
#pragma link C++ class ROOT::Minuit2::FCNGradientBase;
#pragma link C++ class ROOT::Minuit2::FumiliFCNBase;


#endif
