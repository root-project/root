/* @(#)root/physics:$Name:  $:$Id: LinkDef.h,v 1.1.1.1 2000/05/16 17:00:46 rdm Exp $ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class TLorentzVector-;
#pragma link C++ function operator * (Double_t , const TLorentzVector &);

#pragma link C++ class TLorentzRotation+;

#pragma link C++ class TVector3-;
#pragma link C++ function operator + ( const TVector3 &, const TVector3 & );
#pragma link C++ function operator - ( const TVector3 &, const TVector3 & );
#pragma link C++ function operator * ( const TVector3 &, const TVector3 & );
#pragma link C++ function operator * ( const TVector3 &, Double_t  );
#pragma link C++ function operator * ( Double_t, const TVector3 & );
#pragma link C++ function operator * ( const TMatrix &, const TVector3 &);

#pragma link C++ class TVector2-;

#pragma link C++ class TRotation+;
#pragma link C++ class TGenPhaseSpace+;

#endif

