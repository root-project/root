// @(#)root/minuit2:$Name:  $:$Id: LinkDef.hv 1.0 2005/06/23 12:00:00 moneta Exp $
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

#endif
