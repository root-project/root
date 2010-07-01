// @(#)root/eg:$Id$
// Author: Ola Nordmann   29/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAttParticle                                                         //
//                                                                      //
// Particle definition, partly based on GEANT3 particle definition      //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TAttParticle.h"
#include "THashList.h"
#include "TMath.h"
#include "TRandom.h"

ClassImp(TAttParticle)

THashList *TAttParticle::fgList = new THashList;

//______________________________________________________________________________
TAttParticle::TAttParticle()
{
//
//  Particle definition default constructor
//

   //do nothing just set some dummy values
   fPDGMass       = 0.0;
   fPDGStable     = kTRUE;
   fPDGDecayWidth = 0.0;
   fPDGCharge     = 0.0;
   fParticleType  = "";
   fMCnumberOfPDG = 0;
   fEnergyCut     = 1.e-5;
   fEnergyLimit   = 1.e4;
   fGranularity   = 90;
}

//______________________________________________________________________________
TAttParticle::TAttParticle(const char *name, const char *title,
              Double_t Mass, Bool_t Stable,
              Double_t DecayWidth, Double_t Charge, const char *Type,
              Int_t MCnumber, Int_t granularity, Double_t LowerCutOff,
              Double_t HighCutOff) : TNamed(name,title)
{
//
//  Particle definition normal constructor. If the particle is set to be
//  stable, the decay width parameter does have no meaning and can be set to
//  any value. The parameters granularity, LowerCutOff and HighCutOff are
//  used for the construction of the mean free path look up tables. The
//  granularity will be the number of logwise energy points for which the
//  mean free path will be calculated.
//

   fPDGMass       = Mass;
   fPDGStable     = Stable;
   fPDGDecayWidth = DecayWidth;
   fPDGCharge     = Charge;
   fParticleType  = Type;
   fMCnumberOfPDG = MCnumber;
   fEnergyCut     = LowerCutOff;
   fEnergyLimit   = HighCutOff;
   fGranularity   = granularity;

   fgList->Add(this);
}

//______________________________________________________________________________
TAttParticle::~TAttParticle()
{
//
//  Particle destructor
//

}

//______________________________________________________________________________
Int_t TAttParticle::ConvertISAtoPDG(Int_t isaNumber)
{
//
//  Converts the ISAJET Particle number into the PDG MC number
//
   switch (isaNumber) {
      case     1 : return     2; //     UP        .30000E+00       .67
      case    -1 : return    -2; //     UB        .30000E+00      -.67
      case     2 : return     1; //     DN        .30000E+00      -.33
      case    -2 : return    -1; //     DB        .30000E+00       .33
      case     3 : return     3; //     ST        .50000E+00      -.33
      case    -3 : return    -3; //     SB        .50000E+00       .33
      case     4 : return     4; //     CH        .16000E+01       .67
      case    -4 : return    -4; //     CB        .16000E+01      -.67
      case     5 : return     5; //     BT        .49000E+01      -.33
      case    -5 : return    -5; //     BB        .49000E+01       .33
      case     6 : return     7; //     TP        .30000E+02       .67
      case    -6 : return    -7; //     TB        .30000E+02      -.67
      case     9 : return    21; //     GL       0.               0.00
      case    80 : return    24; //     W+        SIN2W=.23       1.00
      case   -80 : return   -24; //     W-        SIN2W=.23      -1.00
      case    90 : return    23; //     Z0        SIN2W=.23       0.00
      case   230 : return   311; //     K0        .49767E+00      0.00
      case  -230 : return  -311; //     AK0       .49767E+00      0.00
      case   330 : return   331; //     ETAP      .95760E+00      0.00
      case   340 : return     0; //     F-        .20300E+01     -1.00
      case  -340 : return     0; //     F+        .20300E+01      1.00
      case   440 : return   441; //     ETAC      .29760E+01      0.00
      case   111 : return   113; //     RHO0      .77000E+00      0.00
      case   121 : return   213; //     RHO+      .77000E+00      1.00
      case  -121 : return  -213; //     RHO-      .77000E+00     -1.00
      case   221 : return   223; //     OMEG      .78260E+00      0.00
      case   131 : return   323; //     K*+       .88810E+00      1.00
      case  -131 : return  -323; //     K*-       .88810E+00     -1.00
      case   231 : return   313; //     K*0       .89220E+00      0.00
      case  -231 : return  -313; //     AK*0      .89220E+00      0.00
      case   331 : return   333; //     PHI       .10196E+01      0.00
      case   141 : return  -423; //     AD*0      .20060E+01      0.00
      case  -141 : return   423; //     D*0       .20060E+01      0.00
      case   241 : return  -413; //     D*-       .20086E+01     -1.00
      case  -241 : return   413; //     D*+       .20086E+01      1.00
      case   341 : return     0; //     F*-       .21400E+01     -1.00
      case  -341 : return     0; //     F*+       .21400E+01      1.00
      case   441 : return   443; //     JPSI      .30970E+01      0.00
      case  1140 : return  4222; //     SC++      .24300E+01      2.00
      case -1140 : return -4222; //     ASC--     .24300E+01     -2.00
      case  1240 : return  4212; //     SC+       .24300E+01      1.00
      case -1240 : return -4212; //     ASC-      .24300E+01     -1.00
      case  2140 : return  4122; //     LC+       .22600E+01      1.00
      case -2140 : return -4122; //     ALC-      .22600E+01     -1.00
      case  2240 : return  4112; //     SC0       .24300E+01      0.00
      case -2240 : return -4112; //     ASC0      .24300E+01      0.00
      case  1340 : return     0; //     USC.      .25000E+01      1.00
      case -1340 : return     0; //     AUSC.     .25000E+01     -1.00
      case  3140 : return     0; //     SUC.      .24000E+01      1.00
      case -3140 : return     0; //     ASUC.     .24000E+01     -1.00
      case  2340 : return     0; //     DSC.      .25000E+01      0.00
      case -2340 : return     0; //     ADSC.     .25000E+01      0.00
      case  3240 : return     0; //     SDC.      .24000E+01      0.00
      case -3240 : return     0; //     ASDC.     .24000E+01      0.00
      case  3340 : return     0; //     SSC.      .26000E+01      0.00
      case -3340 : return     0; //     ASSC.     .26000E+01      0.00
      case  1440 : return     0; //     UCC.      .35500E+01      2.00
      case -1440 : return     0; //     AUCC.     .35500E+01     -2.00
      case  2440 : return     0; //     DCC.      .35500E+01      1.00
      case -2440 : return     0; //     ADCC.     .35500E+01     -1.00
      case  3440 : return     0; //     SCC.      .37000E+01      1.00
      case -3440 : return     0; //     ASCC.     .37000E+01     -1.00
      case  1111 : return  2224; //     DL++      .12320E+01      2.00
      case -1111 : return -2224; //     ADL--     .12320E+01     -2.00
      case  1121 : return  2214; //     DL+       .12320E+01      1.00
      case -1121 : return -2214; //     ADL-      .12320E+01     -1.00
      case  1221 : return  2114; //     DL0       .12320E+01      0.00
      case -1221 : return -2114; //     ADL0      .12320E+01      0.00
      case  2221 : return   1114; //     DL-       .12320E+01     -1.00
      case -2221 : return -1114; //     ADL+      .12320E+01      1.00
      case  1131 : return  3224; //     S*+       .13823E+01      1.00
      case -1131 : return -3224; //     AS*-      .13823E+01     -1.00
      case  1231 : return  3214; //     S*0       .13820E+01      0.00
      case -1231 : return -3214; //     AS*0      .13820E+01      0.00
      case  2231 : return  3114; //     S*-       .13875E+01     -1.00
      case -2231 : return -3114; //     AS*+      .13875E+01      1.00
      case  1331 : return  3324; //     XI*0      .15318E+01      0.00
      case -1331 : return -3324; //     AXI*0     .15318E+01      0.00
      case  2331 : return  3314; //     XI*-      .15350E+01     -1.00
      case -2331 : return -3314; //     AXI*+     .15350E+01      1.00
      case  3331 : return  3334; //     OM-       .16722E+01     -1.00
      case -3331 : return -3334; //     AOM+      .16722E+01      1.00
      case  1141 : return     0; //     UUC*      .26300E+01      2.00
      case -1141 : return     0; //     AUUC*     .26300E+01     -2.00
      case  1241 : return     0; //     UDC*      .26300E+01      1.00
      case -1241 : return     0; //     AUDC*     .26300E+01     -1.00
      case  2241 : return     0; //     DDC*      .26300E+01      0.00
      case -2241 : return     0; //     ADDC*     .26300E+01      0.00
      case  1341 : return     0; //     USC*      .27000E+01      1.00
      case -1341 : return     0; //     AUSC*     .27000E+01     -1.00
      case  2341 : return     0; //     DSC*      .27000E+01      0.00
      case -2341 : return     0; //     ADSC*     .27000E+01      0.00
      case  3341 : return     0; //     SSC*      .28000E+01      0.00
      case -3341 : return     0; //     ASSC*     .28000E+01      0.00
      case  1441 : return     0; //     UCC*      .37500E+01      2.00
      case -1441 : return     0; //     AUCC*     .37500E+01     -2.00
      case  2441 : return     0; //     DCC*      .37500E+01      1.00
      case -2441 : return     0; //     ADCC*     .37500E+01     -1.00
      case  3441 : return     0; //     SCC*      .39000E+01      1.00
      case -3441 : return     0; //     ASCC*     .39000E+01     -1.00
      case  4441 : return     0; //     CCC*      .48000E+01      2.00
      case -4441 : return     0; //     ACCC*     .48000E+01     -2.00
      case    10 : return    22; // Photon
      case    12 : return    11; // Electron
      case   -12 : return   -11; // Positron
      case    14 : return    13; // Muon-
      case   -14 : return   -13; // Muon+
      case    16 : return    15; // Tau-
      case   -16 : return   -15; // Tau+
      case    11 : return    12; // Neutrino e
      case   -11 : return   -12; // Anti Neutrino e
      case    13 : return    14; // Neutrino Muon
      case   -13 : return   -14; // Anti Neutrino Muon
      case    15 : return    16; // Neutrino Tau
      case   -15 : return   -16; // Anti Neutrino Tau
      case   110 : return   111; // Pion0
      case   120 : return   211; // Pion+
      case  -120 : return  -211; // Pion-
      case   220 : return   221; // Eta
      case   130 : return   321; // Kaon+
      case  -130 : return  -321; // Kaon-
      case   -20 : return   130; // Kaon Long
      case    20 : return   310; // Kaon Short
      case  -240 : return  -411; // D+
      case   240 : return   411; // D-
      case  -140 : return   421; // D0
      case   140 : return  -421; // D0 bar
      case  1120 : return  2212; // Proton
      case -1120 : return -2212; // Anti Proton
      case  1220 : return  2112; // Neutron
      case -1220 : return -2112; // Anti Neutron
      case  2130 : return  3122; // Lambda
      case -2130 : return -3122; // Lambda bar
      case  1130 : return  3222; // Sigma+
      case -1130 : return -3222; // Sigma bar -
      case  1230 : return  3212; // Sigma0
      case -1230 : return -3212; // Sigma bar 0
      case  2230 : return  3112; // Sigma-
      case -2230 : return -3112; // Sigma bar +
      case  1330 : return  3322; // Xi0
      case -1330 : return -3322; // Xi bar 0
      case  2330 : return  3312; // Xi-
      case -2330 : return -3312; // Xi bar +
      default :    return 0;      // isajet or pdg number does not exist
   }
}

//______________________________________________________________________________
void TAttParticle::DefinePDG()
{
//
//  Defines particles according to the Particle Data Group
//
//  For questions regarding distribution or content of the MC particle
//  codes, contact
//  Gary Wagman (GSWagman@LBL.BITNET, LBL::GSWagman, or GSWagman@LBL.GOV).
//  (510)486-6610
//
   if (!fgList->IsEmpty()) return;

   new TAttParticle("down","Q001",0.005,kTRUE, .0,-0.333333333333333,"Quark", 1);
   new TAttParticle("down bar","Q001",
                                      0.005,kTRUE, .0,
                                     0.333333333333333,"Quark", -1);
   new TAttParticle("up","Q002",
                                      0.003,kTRUE, .0,
                                     0.666666666666666,"Quark", 2);
   new TAttParticle("up bar","Q002",
                                      0.003,kTRUE, .0,
                                     -0.666666666666666,"Quark", -2);
   new TAttParticle("strange","Q003",
                                      0.1,kTRUE, .0,
                                     -0.333333333333333,"Quark", 3);
   new TAttParticle("strange bar","Q003",
                                      0.1,kTRUE, .0,
                                     0.333333333333333,"Quark", -3);
   new TAttParticle("charm","Q004",
                                      1.4,kTRUE, .0,
                                     0.666666666666666,"Quark", 4);
   new TAttParticle("charm bar","Q004",
                                      1.4,kTRUE, .0,
                                     -0.666666666666666,"Quark", -4);
   new TAttParticle("bottom","Q005",
                                      4.4,kTRUE, .0,
                                     -0.333333333333333,"Quark", 5);
   new TAttParticle("bottom bar","Q005",
                                      4.4,kTRUE, .0,
                                     0.333333333333333,"Quark", -5);
   new TAttParticle("top","Q007",
                                      173.8,kTRUE, .0,
                                     0.666666666666666,"Quark", 6);
   new TAttParticle("top bar","Q007",
                                      173.8,kTRUE, .0,
                                     -0.666666666666666,"Quark", -6);
   new TAttParticle("gluon","G021",
                                      .0,kTRUE, .0,
                                     0.0,"Gauge Boson", 21);
// Entry point of the pdg table conversion
   new TAttParticle("Searches0","S054",
                                      169.0,kTRUE, .0,
                                     0.,"Meson", 7);
   new TAttParticle("e-","S003",
                                      5.10999E-04,kTRUE, .0,
                                     -1.,"Lepton", 11);
   new TAttParticle("e+","S003",
                                      5.10999E-04,kTRUE, .0,
                                     1.,"Lepton", -11);
   new TAttParticle("nu(e)","S001",
                                      .0,kTRUE, .0,
                                     0.0,"Lepton", 12);
   new TAttParticle("nu(e) bar","S001",
                                      .0,kTRUE, .0,
                                     0.0,"Lepton", -12);
   new TAttParticle("mu-","S004",
                                      .1056583,kFALSE, 2.99591E-19,
                                     -1.,"Lepton", 13);
   new TAttParticle("mu+","S004",
                                      .1056583,kFALSE, 2.99591E-19,
                                     1.,"Lepton", -13);
   new TAttParticle("nu(mu)","S002",
                                      .0,kTRUE, .0,
                                     0.0,"Lepton", 14);
   new TAttParticle("nu(mu) bar","S002",
                                      .0,kTRUE, .0,
                                     0.0,"Lepton", -14);
   new TAttParticle("tau-","S035",
                                      1.7771,kFALSE, 2.22700E-12,
                                     -1.,"Lepton", 15);
   new TAttParticle("tau+","S035",
                                      1.7771,kFALSE, 2.22700E-12,
                                     1.,"Lepton", -15);
   new TAttParticle("nu(tau)","S036",
                                      .0,kTRUE, .0,
                                     0.0,"Lepton", 16);
   new TAttParticle("nu(tau) bar","S036",
                                      .0,kTRUE, .0,
                                     0.0,"Lepton", -16);
   new TAttParticle("gamma","S000",
                                      .0,kTRUE, .0,
                                     0.0,"Gauge Boson", 22);
   new TAttParticle("Z0","S044",
                                      91.18699,kFALSE, 2.49,
                                     0.0,"Gauge Boson", 23);
   new TAttParticle("W+","S043",
                                      80.41,kFALSE, 2.06,
                                     +1.,"Gauge Boson", 24);
   new TAttParticle("W-","S043",
                                      80.41,kFALSE, 2.06,
                                     -1.,"Gauge Boson", -24);
   new TAttParticle("pi0","S009",
                                      .1349764,kFALSE, 7.80000E-09,
                                     0.0,"Meson", 111);
   new TAttParticle("rho(770)0","M009",
                                      .7699,kFALSE, .1511999,
                                     0.0,"Meson", 113);
   new TAttParticle("a(2)(1320)0","M012",
                                      1.3181,kFALSE, .107,
                                     0.0,"Meson", 115);
   new TAttParticle("rho(3)(1690)0","M015",
                                      1.691,kFALSE, .160,
                                     0.0,"Meson", 117);
   new TAttParticle("K(L)0","S013",
                                      .4976719,kFALSE, 1.27400E-17,
                                     0.0,"Meson", 130);
   new TAttParticle("pi+","S008",
                                      .1395699,kFALSE, 2.52860E-17,
                                     1.,"Meson", 211);
   new TAttParticle("pi-","S008",
                                      .1395699,kFALSE, 2.52860E-17,
                                     -1.,"Meson", -211);
   new TAttParticle("rho(770)+","M009",
                                      .7699,kFALSE, .1507,
                                     1.,"Meson", 213);
   new TAttParticle("rho(770)-","M009",
                                      .7699,kFALSE, .1507,
                                     -1.,"Meson", -213);
   new TAttParticle("a(2)(1320)+","M012",
                                      1.3181,kFALSE, .107,
                                     1.,"Meson", 215);
   new TAttParticle("a(2)(1320)-","M012",
                                      1.3181,kFALSE, .107,
                                     -1.,"Meson", -215);
   new TAttParticle("rho(3)(1690)+","M015",
                                      1.691,kFALSE, .160,
                                     1.,"Meson", 217);
   new TAttParticle("rho(3)(1690)-","M015",
                                      1.691,kFALSE, .160,
                                     -1.,"Meson", -217);
   new TAttParticle("eta0","S014",
                                      .54730,kFALSE, 1.20000E-06,
                                     0.0,"Meson", 221);
   new TAttParticle("omega(782)0","M001",
                                      .78194,kFALSE, 8.43000E-03,
                                     0.0,"Meson", 223);
   new TAttParticle("f(2)(1270)0","M005",
                                      1.275,kFALSE, .1855,
                                     0.0,"Meson", 225);
   new TAttParticle("omega(3)(1670)0","M045",
                                      1.667,kFALSE, .168,
                                     0.0,"Meson", 227);
   new TAttParticle("f(4)(2050)0","M016",
                                      2.044,kFALSE, .208,
                                     0.0,"Meson", 229);
   new TAttParticle("K(S)0","S012",
                                      .497672,kFALSE, 7.37400E-15,
                                     0.0,"Meson", 310);
   new TAttParticle("K0","S011",
                                      .497672,kFALSE, .0,
                                     0.0,"Meson", 311);
   new TAttParticle("K0 bar","S011",
                                      .497672,kFALSE, .0,
                                     0.0,"Meson", -311);
   new TAttParticle("K*(892)0","M018",
                                      .89610,kFALSE, 5.05000E-02,
                                     0.0,"Meson", 313);
   new TAttParticle("K*(892)0 bar","M018",
                                      .89610,kFALSE, 5.05000E-02,
                                     0.0,"Meson", -313);
   new TAttParticle("K(2)*(1430)0","M022",
                                      1.4324,kFALSE, .1089999,
                                     0.0,"Meson", 315);
   new TAttParticle("K(2)*(1430)0 bar","M022",
                                      1.4324,kFALSE, .1089999,
                                     0.0,"Meson", -315);
   new TAttParticle("K(3)*(1780)0","M060",
                                      1.776,kFALSE, .159,
                                     0.0,"Meson", 317);
   new TAttParticle("K(3)*(1780)0 bar","M060",
                                      1.776,kFALSE, .159,
                                     0.0,"Meson", -317);
   new TAttParticle("K(4)*(2045)0","M035",
                                      2.045,kFALSE, .198,
                                     0.0,"Meson", 319);
   new TAttParticle("K(4)*(2045)0 bar","M035",
                                      2.045,kFALSE, .198,
                                     0.0,"Meson", -319);
   new TAttParticle("K+","S010",
                                      .493677,kFALSE, 5.32100E-17,
                                     1.,"Meson", 321);
   new TAttParticle("K-","S010",
                                      .493677,kFALSE, 5.32100E-17,
                                     -1.,"Meson", -321);
   new TAttParticle("K*(892)+","M018",
                                      .8915899,kFALSE, 5.08000E-02,
                                     1.,"Meson", 323);
   new TAttParticle("K*(892)-","M018",
                                      .8915899,kFALSE, 5.08000E-02,
                                     -1.,"Meson", -323);
   new TAttParticle("K(2)*(1430)+","M022",
                                      1.4256,kFALSE, 9.85000E-02,
                                     1.,"Meson", 325);
   new TAttParticle("K(2)*(1430)-","M022",
                                      1.4256,kFALSE, 9.85000E-02,
                                     -1.,"Meson", -325);
   new TAttParticle("K(3)*(1780)+","M060",
                                      1.776,kFALSE, .159,
                                     1.,"Meson", 327);
   new TAttParticle("K(3)*(1780)-","M060",
                                      1.776,kFALSE, .159,
                                     -1.,"Meson", -327);
   new TAttParticle("K(4)*(2045)+","M035",
                                      2.045,kFALSE, .198,
                                     1.,"Meson", 329);
   new TAttParticle("K(4)*(2045)-","M035",
                                      2.045,kFALSE, .198,
                                     -1.,"Meson", -329);
   new TAttParticle("eta'(958)0","M002",
                                      .95778,kFALSE, 2.03000E-04,
                                     0.0,"Meson", 331);
   new TAttParticle("phi(1020)0","M004",
                                      1.01941,kFALSE, 4.43000E-03,
                                     0.0,"Meson", 333);
   new TAttParticle("f(2)'(1525)0","M013",
                                      1.525,kFALSE, 7.60000E-02,
                                     0.0,"Meson", 335);
   new TAttParticle("phi(3)(1850)0","M054",
                                      1.854,kFALSE, 8.70000E-02,
                                     0.0,"Meson", 337);
   new TAttParticle("D+","S031",
                                      1.8693,kFALSE, 6.23000E-13,
                                     1.,"Meson", 411);
   new TAttParticle("D-","S031",
                                      1.8693,kFALSE, 6.23000E-13,
                                     -1.,"Meson", -411);
   new TAttParticle("D*(2010)+","M062",
                                      2.01,kTRUE, .0,
                                     1.,"Meson", 413);
   new TAttParticle("D*(2010)-","M062",
                                      2.01,kTRUE, .0,
                                     -1.,"Meson", -413);
   new TAttParticle("D(2)*(2460)+","M150",
                                      2.4589,kFALSE, 2.30000E-02,
                                     1.,"Meson", 415);
   new TAttParticle("D(2)*(2460)-","M150",
                                      2.4589,kFALSE, 2.30000E-02,
                                     -1.,"Meson", -415);
   new TAttParticle("D0","S032",
                                      1.8646,kFALSE, 1.58600E-12,
                                     0.0,"Meson", 421);
   new TAttParticle("D*(2007)0","M061",
                                      2.0067,kTRUE, .0,
                                     0.0,"Meson", 423);
   new TAttParticle("D(2)*(2460)0","M119",
                                      2.4589,kFALSE, 2.30000E-02,
                                     0.0,"Meson", 425);
   new TAttParticle("D(s)+","S034",
                                      1.9685,kFALSE, 1.41000E-12,
                                     1.,"Meson", 431);
   new TAttParticle("D(s)-","S034",
                                      1.9685,kFALSE, 1.41000E-12,
                                     -1.,"Meson", -431);
   new TAttParticle("D(s)*+","S074",
                                      2.1124,kTRUE, .0,
                                     1.,"Meson", 433);
   new TAttParticle("D(s)*-","S074",
                                      2.1124,kTRUE, .0,
                                     -1.,"Meson", -433);
   new TAttParticle("eta(c)(1S)0","M026",
                                      2.9798,kFALSE, 1.32000E-02,
                                     0.0,"Meson", 441);
   new TAttParticle("J/psi(1S)0","M070",
                                      3.09688,kFALSE, 8.70000E-05,
                                     0.0,"Meson", 443);
   new TAttParticle("chi(c2)(1P)0","M057",
                                      3.55617,kFALSE, 2.00000E-03,
                                     0.0,"Meson", 445);
   new TAttParticle("B0","S049",
                                      5.2792,kFALSE, 4.39000E-13,
                                     0.0,"Meson", 511);
   new TAttParticle("B*0","S085",
                                      5.3249,kTRUE, .0,
                                     0.0,"Meson", 513);
   new TAttParticle("B+","S049",
                                      5.2789,kFALSE, 4.28000E-13,
                                     1.,"Meson", 521);
   new TAttParticle("B-","S049",
                                      5.2789,kFALSE, 4.28000E-13,
                                     -1.,"Meson", -521);
   new TAttParticle("B*+","S085",
                                      5.3249,kTRUE, .0,
                                     1.,"Meson", 523);
   new TAttParticle("B*-","S085",
                                      5.3249,kTRUE, .0,
                                     -1.,"Meson", -523);
   new TAttParticle("B(s)0","S086",
                                      5.3693,kFALSE, 4.90000E-13,
                                     0.0,"Meson", 531);
   new TAttParticle("chi(b0)(1P)0","M076",
                                      9.8598,kTRUE, .0,
                                     0.0,"Meson", 551);
   new TAttParticle("Upsilon(1S)0","M049",
                                      9.46037,kFALSE, 5.25000E-05,
                                     0.0,"Meson", 553);
   new TAttParticle("chi(b2)(1P)0","M078",
                                      9.9132,kTRUE, .0,
                                     0.0,"Meson", 555);
   new TAttParticle("Delta(1620)-","B082",
                                      1.62,kFALSE, .15,
                                     -1.,"Baryon", 1112);
   new TAttParticle("Delta(1620)+ bar","B082",
                                      1.62,kFALSE, .15,
                                     +1.,"Baryon", -1112);
   new TAttParticle("Delta(1232)-","B033",
                                      1.232,kFALSE, .1199999,
                                     -1.,"Baryon", 1114);
   new TAttParticle("Delta(1232)+ bar","B033",
                                      1.232,kFALSE, .1199999,
                                     +1.,"Baryon", -1114);
   new TAttParticle("Delta(1905)-","B011",
                                      1.905,kFALSE, .3499999,
                                     -1.,"Baryon", 1116);
   new TAttParticle("Delta(1905)+ bar","B011",
                                      1.905,kFALSE, .3499999,
                                     +1.,"Baryon", -1116);
   new TAttParticle("Delta(1950)-","B083",
                                      1.95,kFALSE, .3,
                                     -1.,"Baryon", 1118);
   new TAttParticle("Delta(1950)+ bar","B083",
                                      1.95,kFALSE, .3,
                                     +1.,"Baryon", -1118);
   new TAttParticle("Delta(1620)0","B082",
                                      1.62,kFALSE, .15,
                                     0.0,"Baryon", 1212);
   new TAttParticle("Delta(1620)0 bar","B082",
                                      1.62,kFALSE, .15,
                                     0.0,"Baryon", -1212);
   new TAttParticle("N(1520)0","B062",
                                      1.52,kFALSE, .1199999,
                                     0.0,"Baryon", 1214);
   new TAttParticle("N(1520)0 bar","B062",
                                      1.52,kFALSE, .1199999,
                                     0.0,"Baryon", -1214);
   new TAttParticle("Delta(1905)0","B011",
                                      1.905,kFALSE, .3499999,
                                     0.0,"Baryon", 1216);
   new TAttParticle("Delta(1905)0 bar","B011",
                                      1.905,kFALSE, .3499999,
                                     0.0,"Baryon", -1216);
   new TAttParticle("N(2190)0","B071",
                                      2.19,kFALSE, .4499999,
                                     0.0,"Baryon", 1218);
   new TAttParticle("N(2190)0 bar","B071",
                                      2.19,kFALSE, .4499999,
                                     0.0,"Baryon", -1218);
   new TAttParticle("n","S017",
                                      .9395656,kFALSE, 7.42100E-28,
                                     0.0,"Baryon", 2112);
   new TAttParticle("n bar","S017",
                                      .9395656,kFALSE, 7.42100E-28,
                                     0.0,"Baryon", -2112);
   new TAttParticle("Delta(1232)0","B033",
                                      1.232,kFALSE, .1199999,
                                     0.0,"Baryon", 2114);
   new TAttParticle("Delta(1232)0 bar","B033",
                                      1.232,kFALSE, .1199999,
                                     0.0,"Baryon", -2114);
   new TAttParticle("N(1675)0","B064",
                                      1.675,kFALSE, .15,
                                     0.0,"Baryon", 2116);
   new TAttParticle("N(1675)0 bar","B064",
                                      1.675,kFALSE, .15,
                                     0.0,"Baryon", -2116);
   new TAttParticle("Delta(1950)0","B083",
                                      1.95,kFALSE, .3,
                                     0.0,"Baryon", 2118);
   new TAttParticle("Delta(1950)0 bar","B083",
                                      1.95,kFALSE, .3,
                                     0.0,"Baryon", -2118);
   new TAttParticle("Delta(1620)+","B082",
                                      1.62,kFALSE, .15,
                                     +1.,"Baryon", 2122);
   new TAttParticle("Delta(1620)- bar","B082",
                                      1.62,kFALSE, .15,
                                     -1.,"Baryon", -2122);
   new TAttParticle("N(1520)+","B062",
                                      1.52,kFALSE, .1199999,
                                     +1.,"Baryon", 2124);
   new TAttParticle("N(1520)- bar","B062",
                                      1.52,kFALSE, .1199999,
                                     -1.,"Baryon", -2124);
   new TAttParticle("Delta(1905)+","B011",
                                      1.905,kFALSE, .3499999,
                                     +1.,"Baryon", 2126);
   new TAttParticle("Delta(1905)- bar","B011",
                                      1.905,kFALSE, .3499999,
                                     -1.,"Baryon", -2126);
   new TAttParticle("N(2190)+","B071",
                                      2.19,kFALSE, .4499999,
                                     +1.,"Baryon", 2128);
   new TAttParticle("N(2190)- bar","B071",
                                      2.19,kFALSE, .4499999,
                                     -1.,"Baryon", -2128);
   new TAttParticle("p","S016",
                                      .9382722,kTRUE, .0,
                                     +1.,"Baryon", 2212);
   new TAttParticle("p bar","S016",
                                      .9382722,kTRUE, .0,
                                     -1.,"Baryon", -2212);
   new TAttParticle("Delta(1232)+","B033",
                                      1.232,kFALSE, .1199999,
                                     +1.,"Baryon", 2214);
   new TAttParticle("Delta(1232)- bar","B033",
                                      1.232,kFALSE, .1199999,
                                     -1.,"Baryon", -2214);
   new TAttParticle("N(1675)+","B064",
                                      1.675,kFALSE, .15,
                                     +1.,"Baryon", 2216);
   new TAttParticle("N(1675)- bar","B064",
                                      1.675,kFALSE, .15,
                                     -1.,"Baryon", -2216);
   new TAttParticle("Delta(1950)+","B083",
                                      1.95,kFALSE, .3,
                                     +1.,"Baryon", 2218);
   new TAttParticle("Delta(1950)- bar","B083",
                                      1.95,kFALSE, .3,
                                     -1.,"Baryon", -2218);
   new TAttParticle("Delta(1620)++","B082",
                                      1.62,kFALSE, .15,
                                     +2.,"Baryon", 2222);
   new TAttParticle("Delta(1620)-- bar","B082",
                                      1.62,kFALSE, .15,
                                     -2.,"Baryon", -2222);
   new TAttParticle("Delta(1232)++","B033",
                                      1.232,kFALSE, .1199999,
                                     +2.,"Baryon", 2224);
   new TAttParticle("Delta(1232)-- bar","B033",
                                      1.232,kFALSE, .1199999,
                                     -2.,"Baryon", -2224);
   new TAttParticle("Delta(1905)++","B011",
                                      1.905,kFALSE, .3499999,
                                     +2.,"Baryon", 2226);
   new TAttParticle("Delta(1905)-- bar","B011",
                                      1.905,kFALSE, .3499999,
                                     -2.,"Baryon", -2226);
   new TAttParticle("Delta(1950)++","B083",
                                      1.95,kFALSE, .3,
                                     +2.,"Baryon", 2228);
   new TAttParticle("Delta(1950)-- bar","B083",
                                      1.95,kFALSE, .3,
                                     -2.,"Baryon", -2228);
   new TAttParticle("Sigma-","S020",
                                      1.19744,kFALSE, 4.45000E-15,
                                     -1.,"Baryon", 3112);
   new TAttParticle("Sigma+ bar","S020",
                                      1.19744,kFALSE, 4.45000E-15,
                                     +1.,"Baryon", -3112);
   new TAttParticle("Sigma(1385)-","B043",
                                      1.3872,kFALSE, 3.94000E-02,
                                     -1.,"Baryon", 3114);
   new TAttParticle("Sigma(1385)+ bar","B043",
                                      1.3872,kFALSE, 3.94000E-02,
                                     +1.,"Baryon", -3114);
   new TAttParticle("Sigma(1775)-","B045",
                                      1.775,kFALSE, .1199999,
                                     -1.,"Baryon", 3116);
   new TAttParticle("Sigma(1775)+ bar","B045",
                                      1.775,kFALSE, .1199999,
                                     +1.,"Baryon", -3116);
   new TAttParticle("Sigma(2030)-","B047",
                                      2.03,kFALSE, .18,
                                     -1.,"Baryon", 3118);
   new TAttParticle("Sigma(2030)+ bar","B047",
                                      2.03,kFALSE, .18,
                                     +1.,"Baryon", -3118);
   new TAttParticle("Lambda0","S018",
                                      1.11568,kFALSE, 2.50100E-15,
                                     0.0,"Baryon", 3122);
   new TAttParticle("Lambda0 bar","S018",
                                      1.11568,kFALSE, 2.50100E-15,
                                     0.0,"Baryon", -3122);
   new TAttParticle("Lambda(1520)0","B038",
                                      1.5195,kFALSE, 1.56000E-02,
                                     0.0,"Baryon", 3124);
   new TAttParticle("Lambda(1520)0 bar","B038",
                                      1.5195,kFALSE, 1.56000E-02,
                                     0.0,"Baryon", -3124);
   new TAttParticle("Lambda(1820)0","B039",
                                      1.82,kFALSE, 8.00000E-02,
                                     0.0,"Baryon", 3126);
   new TAttParticle("Lambda(1820)0 bar","B039",
                                      1.82,kFALSE, 8.00000E-02,
                                     0.0,"Baryon", -3126);
   new TAttParticle("Lambda(2100)0","B041",
                                      2.1,kFALSE, .2,
                                     0.0,"Baryon", 3128);
   new TAttParticle("Lambda(2100)0 bar","B041",
                                      2.1,kFALSE, .2,
                                     0.0,"Baryon", -3128);
   new TAttParticle("Sigma0","S021",
                                      1.19255,kFALSE, 8.90000E-06,
                                     0.0,"Baryon", 3212);
   new TAttParticle("Sigma0 bar","S021",
                                      1.19255,kFALSE, 8.90000E-06,
                                     0.0,"Baryon", -3212);
   new TAttParticle("Sigma(1385)0","B043",
                                      1.3837,kFALSE, 3.60000E-02,
                                     0.0,"Baryon", 3214);
   new TAttParticle("Sigma(1385)0 bar","B043",
                                      1.3837,kFALSE, 3.60000E-02,
                                     0.0,"Baryon", -3214);
   new TAttParticle("Sigma(1775)0","B045",
                                      1.775,kFALSE, .1199999,
                                     0.0,"Baryon", 3216);
   new TAttParticle("Sigma(1775)0 bar","B045",
                                      1.775,kFALSE, .1199999,
                                     0.0,"Baryon", -3216);
   new TAttParticle("Sigma(2030)0","B047",
                                      2.03,kFALSE, .18,
                                     0.0,"Baryon", 3218);
   new TAttParticle("Sigma(2030)0 bar","B047",
                                      2.03,kFALSE, .18,
                                     0.0,"Baryon", -3218);
   new TAttParticle("Sigma+","S019",
                                      1.18937,kFALSE, 8.24000E-15,
                                     +1.,"Baryon", 3222);
   new TAttParticle("Sigma- bar","S019",
                                      1.18937,kFALSE, 8.24000E-15,
                                     -1.,"Baryon", -3222);
   new TAttParticle("Sigma(1385)+","B043",
                                      1.3828,kFALSE, 3.58000E-02,
                                     +1.,"Baryon", 3224);
   new TAttParticle("Sigma(1385)- bar","B043",
                                      1.3828,kFALSE, 3.58000E-02,
                                     -1.,"Baryon", -3224);
   new TAttParticle("Sigma(1775)+","B045",
                                      1.775,kFALSE, .1199999,
                                     +1.,"Baryon", 3226);
   new TAttParticle("Sigma(1775)- bar","B045",
                                      1.775,kFALSE, .1199999,
                                     -1.,"Baryon", -3226);
   new TAttParticle("Sigma(2030)+","B047",
                                      2.03,kFALSE, .18,
                                     +1.,"Baryon", 3228);
   new TAttParticle("Sigma(2030)- bar","B047",
                                      2.03,kFALSE, .18,
                                     -1.,"Baryon", -3228);
   new TAttParticle("Xi-","S022",
                                      1.32132,kFALSE, 4.02000E-15,
                                     -1.,"Baryon", 3312);
   new TAttParticle("Xi+ bar","S022",
                                      1.32132,kFALSE, 4.02000E-15,
                                     +1.,"Baryon", -3312);
   new TAttParticle("Xi(1530)-","B049",
                                      1.535,kFALSE, 9.90000E-03,
                                     -1.,"Baryon", 3314);
   new TAttParticle("Xi(1530)+ bar","B049",
                                      1.535,kFALSE, 9.90000E-03,
                                     +1.,"Baryon", -3314);
   new TAttParticle("Xi0","S023",
                                      1.3149,kFALSE, 2.27000E-15,
                                     0.0,"Baryon", 3322);
   new TAttParticle("Xi0 bar","S023",
                                      1.3149,kFALSE, 2.27000E-15,
                                     0.0,"Baryon", -3322);
   new TAttParticle("Xi(1530)0","B049",
                                      1.5318,kFALSE, 9.10000E-03,
                                     0.0,"Baryon", 3324);
   new TAttParticle("Xi(1530)0 bar","B049",
                                      1.5318,kFALSE, 9.10000E-03,
                                     0.0,"Baryon", -3324);
   new TAttParticle("Omega-","S024",
                                      1.67245,kFALSE, 8.01000E-15,
                                     -1.,"Baryon", 3334);
   new TAttParticle("Omega+ bar","S024",
                                      1.67245,kFALSE, 8.01000E-15,
                                     +1.,"Baryon", -3334);
   new TAttParticle("Sigma(c)(2455)0","B104",
                                      2.4524,kTRUE, .0,
                                     0.0,"Baryon", 4112);
   new TAttParticle("Sigma(c)(2455)0 bar","B104",
                                      2.4524,kTRUE, .0,
                                     0.0,"Baryon", -4112);
   new TAttParticle("Lambda(c)+","S033",
                                      2.2849,kFALSE, 3.29000E-12,
                                     +1.,"Baryon", 4122);
   new TAttParticle("Lambda(c)- bar","S033",
                                      2.2849,kFALSE, 3.29000E-12,
                                     -1.,"Baryon", -4122);
   new TAttParticle("Sigma(c)(2455)+","B104",
                                      2.4538,kTRUE, .0,
                                     +1.,"Baryon", 4212);
   new TAttParticle("Sigma(c)(2455)- bar","B104",
                                      2.4538,kTRUE, .0,
                                     -1.,"Baryon", -4212);
   new TAttParticle("Sigma(c)(2455)++","B104",
                                      2.4531,kTRUE, .0,
                                     +2.,"Baryon", 4222);
   new TAttParticle("Sigma(c)(2455)-- bar","B104",
                                      2.4531,kTRUE, .0,
                                     -2.,"Baryon", -4222);
   new TAttParticle("Xi(c)0","S048",
                                      2.4703,kFALSE, 6.70000E-12,
                                     0.0,"Baryon", 4312);
   new TAttParticle("Xi(c)0 bar","S048",
                                      2.4703,kFALSE, 6.70000E-12,
                                     0.0,"Baryon", -4312);
   new TAttParticle("Xi(c)+","S045",
                                      2.4651,kFALSE, 1.86000E-12,
                                     +1.,"Baryon", 4322);
   new TAttParticle("Xi(c)- bar","S045",
                                      2.4651,kFALSE, 1.86000E-12,
                                     -1.,"Baryon", -4322);
   new TAttParticle("Lambda(b)0","S040",
                                      5.64,kFALSE, 6.20000E-13,
                                     0.0,"Baryon", 5122);
   new TAttParticle("Lambda(b)0 bar","S040",
                                      5.64,kFALSE, 6.20000E-13,
                                     0.0,"Baryon", -5122);
   new TAttParticle("a(0)(980)0","M036",
                                      .9824,kTRUE, .0,
                                     0.0,"Meson", 10111);
   new TAttParticle("b(1)(1235)0","M011",
                                      1.231,kFALSE, .142,
                                     0.0,"Meson", 10113);
   new TAttParticle("pi(2)(1670)0","M034",
                                      1.67,kFALSE, .2399999,
                                     0.0,"Meson", 10115);
   new TAttParticle("a(0)(980)+","M036",
                                      .9834,kTRUE, .0,
                                     1.,"Meson", 10211);
   new TAttParticle("a(0)(980)-","M036",
                                      .9834,kTRUE, .0,
                                     -1.,"Meson", -10211);
   new TAttParticle("b(1)(1235)+","M011",
                                      1.2295,kFALSE, .142,
                                     1.,"Meson", 10213);
   new TAttParticle("b(1)(1235)-","M011",
                                      1.2295,kFALSE, .142,
                                     -1.,"Meson", -10213);
   new TAttParticle("pi(2)(1670)+","M034",
                                      1.67,kFALSE, .2399999,
                                     1.,"Meson", 10215);
   new TAttParticle("pi(2)(1670)-","M034",
                                      1.67,kFALSE, .2399999,
                                     -1.,"Meson", -10215);
   new TAttParticle("f(0)(980)0","M003",
                                      .98,kTRUE, .0,
                                     0.0,"Meson", 10221);
   new TAttParticle("h(1)(1170)0","M030",
                                      1.17,kFALSE, .36,
                                     0.0,"Meson", 10223);
   new TAttParticle("K(0)*(1430)0","M019",
                                      1.429,kFALSE, .287,
                                     0.0,"Meson", 10311);
   new TAttParticle("K(0)*(1430)0 bar","M019",
                                      1.429,kFALSE, .287,
                                     0.0,"Meson", -10311);
   new TAttParticle("K(1)(1270)0","M028",
                                      1.272,kFALSE, 9.00000E-02,
                                     0.0,"Meson", 10313);
   new TAttParticle("K(1)(1270)0 bar","M028",
                                      1.272,kFALSE, 9.00000E-02,
                                     0.0,"Meson", -10313);
   new TAttParticle("K(2)(1770)0","M023",
                                      1.773,kFALSE, .186,
                                     0.0,"Meson", 10315);
   new TAttParticle("K(2)(1770)0 bar","M023",
                                      1.773,kFALSE, .186,
                                     0.0,"Meson", -10315);
   new TAttParticle("K(0)*(1430)+","M019",
                                      1.429,kFALSE, .287,
                                     1.,"Meson", 10321);
   new TAttParticle("K(0)*(1430)-","M019",
                                      1.429,kFALSE, .287,
                                     -1.,"Meson", -10321);
   new TAttParticle("K(1)(1270)+","M028",
                                      1.272,kFALSE, 9.00000E-02,
                                     1.,"Meson", 10323);
   new TAttParticle("K(1)(1270)-","M028",
                                      1.272,kFALSE, 9.00000E-02,
                                     -1.,"Meson", -10323);
   new TAttParticle("K(2)(1770)+","M023",
                                      1.773,kFALSE, .186,
                                     1.,"Meson", 10325);
   new TAttParticle("K(2)(1770)-","M023",
                                      1.773,kFALSE, .186,
                                     -1.,"Meson", -10325);
   new TAttParticle("phi(1680)0","M067",
                                      1.68,kFALSE, .15,
                                     0.0,"Meson", 10333);
   new TAttParticle("D(1)(2420)0","M097",
                                      2.4228,kFALSE, 1.80000E-02,
                                     0.0,"Meson", 10423);
   new TAttParticle("D(s1)(2536)+","M121",
                                      2.53535,kTRUE, .0,
                                     1.,"Meson", 10433);
   new TAttParticle("D(s1)(2536)-","M121",
                                      2.53535,kTRUE, .0,
                                     -1.,"Meson", -10433);
   new TAttParticle("chi(c0)(1P)0","M056",
                                      3.4151,kFALSE, 1.40000E-02,
                                     0.0,"Meson", 10441);
   new TAttParticle("chi(c1)(1P)0","M055",
                                      3.51053,kFALSE, 8.80000E-04,
                                     0.0,"Meson", 10443);
   new TAttParticle("chi(b0)(2P)0","M079",
                                      10.23209,kTRUE, .0,
                                     0.0,"Meson", 10551);
   new TAttParticle("chi(b1)(1P)0","M077",
                                      9.8919,kTRUE, .0,
                                     0.0,"Meson", 10553);
   new TAttParticle("chi(b2)(2P)0","M081",
                                      10.2685,kTRUE, .0,
                                     0.0,"Meson", 10555);
   new TAttParticle("Delta(1900)-","B030",
                                      1.9,kFALSE, .2,
                                     -1.,"Baryon", 11112);
   new TAttParticle("Delta(1900)+ bar","B030",
                                      1.9,kFALSE, .2,
                                     +1.,"Baryon", -11112);
   new TAttParticle("Delta(1700)-","B010",
                                      1.7,kFALSE, .3,
                                     -1.,"Baryon", 11114);
   new TAttParticle("Delta(1700)+ bar","B010",
                                      1.7,kFALSE, .3,
                                     +1.,"Baryon", -11114);
   new TAttParticle("Delta(1930)-","B013",
                                      1.93,kFALSE, .3499999,
                                     -1.,"Baryon", 11116);
   new TAttParticle("Delta(1930)+ bar","B013",
                                      1.93,kFALSE, .3499999,
                                     +1.,"Baryon", -11116);
   new TAttParticle("Delta(1900)0","B030",
                                      1.9,kFALSE, .2,
                                     0.0,"Baryon", 11212);
   new TAttParticle("Delta(1900)0 bar","B030",
                                      1.9,kFALSE, .2,
                                     0.0,"Baryon", -11212);
   new TAttParticle("Delta(1930)0","B013",
                                      1.93,kFALSE, .3499999,
                                     0.0,"Baryon", 11216);
   new TAttParticle("Delta(1930)0 bar","B013",
                                      1.93,kFALSE, .3499999,
                                     0.0,"Baryon", -11216);
   new TAttParticle("N(1440)0","B061",
                                      1.44,kFALSE, .3499999,
                                     0.0,"Baryon", 12112);
   new TAttParticle("N(1440)0 bar","B061",
                                      1.44,kFALSE, .3499999,
                                     0.0,"Baryon", -12112);
   new TAttParticle("Delta(1700)0","B010",
                                      1.7,kFALSE, .3,
                                     0.0,"Baryon", 12114);
   new TAttParticle("Delta(1700)0 bar","B010",
                                      1.7,kFALSE, .3,
                                     0.0,"Baryon", -12114);
   new TAttParticle("N(1680)0","B065",
                                      1.68,kFALSE, .1299999,
                                     0.0,"Baryon", 12116);
   new TAttParticle("N(1680)0 bar","B065",
                                      1.68,kFALSE, .1299999,
                                     0.0,"Baryon", -12116);
   new TAttParticle("Delta(1900)+","B030",
                                      1.9,kFALSE, .2,
                                     +1.,"Baryon", 12122);
   new TAttParticle("Delta(1900)- bar","B030",
                                      1.9,kFALSE, .2,
                                     -1.,"Baryon", -12122);
   new TAttParticle("Delta(1930)+","B013",
                                      1.93,kFALSE, .3499999,
                                     +1.,"Baryon", 12126);
   new TAttParticle("Delta(1930)- bar","B013",
                                      1.93,kFALSE, .3499999,
                                     -1.,"Baryon", -12126);
   new TAttParticle("N(1440)+","B061",
                                      1.44,kFALSE, .3499999,
                                     +1.,"Baryon", 12212);
   new TAttParticle("N(1440)- bar","B061",
                                      1.44,kFALSE, .3499999,
                                     -1.,"Baryon", -12212);
   new TAttParticle("Delta(1700)+","B010",
                                      1.7,kFALSE, .3,
                                     +1.,"Baryon", 12214);
   new TAttParticle("Delta(1700)- bar","B010",
                                      1.7,kFALSE, .3,
                                     -1.,"Baryon", -12214);
   new TAttParticle("N(1680)+","B065",
                                      1.68,kFALSE, .1299999,
                                     +1.,"Baryon", 12216);
   new TAttParticle("N(1680)- bar","B065",
                                      1.68,kFALSE, .1299999,
                                     -1.,"Baryon", -12216);
   new TAttParticle("Delta(1900)++","B030",
                                      1.9,kFALSE, .2,
                                     +2.,"Baryon", 12222);
   new TAttParticle("Delta(1900)-- bar","B030",
                                      1.9,kFALSE, .2,
                                     -2.,"Baryon", -12222);
   new TAttParticle("Delta(1700)++","B010",
                                      1.7,kFALSE, .3,
                                     +2.,"Baryon", 12224);
   new TAttParticle("Delta(1700)-- bar","B010",
                                      1.7,kFALSE, .3,
                                     -2.,"Baryon", -12224);
   new TAttParticle("Delta(1930)++","B013",
                                      1.93,kFALSE, .3499999,
                                     +2.,"Baryon", 12226);
   new TAttParticle("Delta(1930)-- bar","B013",
                                      1.93,kFALSE, .3499999,
                                     -2.,"Baryon", -12226);
   new TAttParticle("Sigma(1660)-","B079",
                                      1.66,kFALSE, .1,
                                     -1.,"Baryon", 13112);
   new TAttParticle("Sigma(1660)+ bar","B079",
                                      1.66,kFALSE, .1,
                                     +1.,"Baryon", -13112);
   new TAttParticle("Sigma(1670)-","B051",
                                      1.67,kFALSE, 6.00000E-02,
                                     -1.,"Baryon", 13114);
   new TAttParticle("Sigma(1670)+ bar","B051",
                                      1.67,kFALSE, 6.00000E-02,
                                     +1.,"Baryon", -13114);
   new TAttParticle("Sigma(1915)-","B046",
                                      1.915,kFALSE, .1199999,
                                     -1.,"Baryon", 13116);
   new TAttParticle("Sigma(1915)+ bar","B046",
                                      1.915,kFALSE, .1199999,
                                     +1.,"Baryon", -13116);
   new TAttParticle("Lambda(1405)0","B037",
                                      1.407,kFALSE, 5.00000E-02,
                                     0.0,"Baryon", 13122);
   new TAttParticle("Lambda(1405)0 bar","B037",
                                      1.407,kFALSE, 5.00000E-02,
                                     0.0,"Baryon", -13122);
   new TAttParticle("Lambda(1690)0","B055",
                                      1.69,kFALSE, 6.00000E-02,
                                     0.0,"Baryon", 13124);
   new TAttParticle("Lambda(1690)0 bar","B055",
                                      1.69,kFALSE, 6.00000E-02,
                                     0.0,"Baryon", -13124);
   new TAttParticle("Lambda(1830)0","B056",
                                      1.83,kFALSE, 9.50000E-02,
                                     0.0,"Baryon", 13126);
   new TAttParticle("Lambda(1830)0 bar","B056",
                                      1.83,kFALSE, 9.50000E-02,
                                     0.0,"Baryon", -13126);
   new TAttParticle("Sigma(1660)0","B079",
                                      1.66,kFALSE, .1,
                                     0.0,"Baryon", 13212);
   new TAttParticle("Sigma(1660)0 bar","B079",
                                      1.66,kFALSE, .1,
                                     0.0,"Baryon", -13212);
   new TAttParticle("Sigma(1670)0","B051",
                                      1.67,kFALSE, 6.00000E-02,
                                     0.0,"Baryon", 13214);
   new TAttParticle("Sigma(1670)0 bar","B051",
                                      1.67,kFALSE, 6.00000E-02,
                                     0.0,"Baryon", -13214);
   new TAttParticle("Sigma(1915)0","B046",
                                      1.915,kFALSE, .1199999,
                                     0.0,"Baryon", 13216);
   new TAttParticle("Sigma(1915)0 bar","B046",
                                      1.915,kFALSE, .1199999,
                                     0.0,"Baryon", -13216);
   new TAttParticle("Sigma(1660)+","B079",
                                      1.66,kFALSE, .1,
                                     +1.,"Baryon", 13222);
   new TAttParticle("Sigma(1660)- bar","B079",
                                      1.66,kFALSE, .1,
                                     -1.,"Baryon", -13222);
   new TAttParticle("Sigma(1670)+","B051",
                                      1.67,kFALSE, 6.00000E-02,
                                     +1.,"Baryon", 13224);
   new TAttParticle("Sigma(1670)- bar","B051",
                                      1.67,kFALSE, 6.00000E-02,
                                     -1.,"Baryon", -13224);
   new TAttParticle("Sigma(1915)+","B046",
                                      1.915,kFALSE, .1199999,
                                     +1.,"Baryon", 13226);
   new TAttParticle("Sigma(1915)- bar","B046",
                                      1.915,kFALSE, .1199999,
                                     -1.,"Baryon", -13226);
   new TAttParticle("Xi(1820)-","B050",
                                      1.823,kFALSE, 2.40000E-02,
                                     -1.,"Baryon", 13314);
   new TAttParticle("Xi(1820)+ bar","B050",
                                      1.823,kFALSE, 2.40000E-02,
                                     +1.,"Baryon", -13314);
   new TAttParticle("Xi(1820)0","B050",
                                      1.823,kFALSE, 2.40000E-02,
                                     0.0,"Baryon", 13324);
   new TAttParticle("Xi(1820)0 bar","B050",
                                      1.823,kFALSE, 2.40000E-02,
                                     0.0,"Baryon", -13324);
   new TAttParticle("pi(1300)0","M058",
                                      1.3,kTRUE, .4,
                                     0.0,"Meson", 20111);
   new TAttParticle("a(1)(1260)0","M010",
                                      1.23,kTRUE, .4,
                                     0.0,"Meson", 20113);
   new TAttParticle("pi(1300)+","M058",
                                      1.3,kTRUE, .4,
                                     1.,"Meson", 20211);
   new TAttParticle("pi(1300)-","M058",
                                      1.3,kTRUE, .4,
                                     -1.,"Meson", -20211);
   new TAttParticle("a(1)(1260)+","M010",
                                      1.23,kTRUE, .4,
                                     1.,"Meson", 20213);
   new TAttParticle("a(1)(1260)-","M010",
                                      1.23,kTRUE, .4,
                                     -1.,"Meson", -20213);
   new TAttParticle("eta(1295)0","M037",
                                      1.297,kFALSE, 5.30000E-02,
                                     0.0,"Meson", 20221);
   new TAttParticle("f(1)(1285)0","M008",
                                      1.282,kFALSE, 2.40000E-02,
                                     0.0,"Meson", 20223);
   new TAttParticle("f(2)(2010)0","M106",
                                      2.01,kFALSE, .2,
                                     0.0,"Meson", 20225);
   new TAttParticle("K(1)(1400)0","M064",
                                      1.402,kFALSE, .1739999,
                                     0.0,"Meson", 20313);
   new TAttParticle("K(1)(1400)0 bar","M064",
                                      1.402,kFALSE, .1739999,
                                     0.0,"Meson", -20313);
   new TAttParticle("K(2)(1820)0","M146",
                                      1.816,kFALSE, .2759999,
                                     0.0,"Meson", 20315);
   new TAttParticle("K(2)(1820)0 bar","M146",
                                      1.816,kFALSE, .2759999,
                                     0.0,"Meson", -20315);
   new TAttParticle("K(1)(1400)+","M064",
                                      1.402,kFALSE, .1739999,
                                     1.,"Meson", 20323);
   new TAttParticle("K(1)(1400)-","M064",
                                      1.402,kFALSE, .1739999,
                                     -1.,"Meson", -20323);
   new TAttParticle("K(2)(1820)+","M146",
                                      1.816,kFALSE, .2759999,
                                     1.,"Meson", 20325);
   new TAttParticle("K(2)(1820)-","M146",
                                      1.816,kFALSE, .2759999,
                                     -1.,"Meson", -20325);
   new TAttParticle("psi(2S)0","M071",
                                      3.686,kFALSE, 2.77000E-04,
                                     0.0,"Meson", 20443);
   new TAttParticle("Upsilon(2S)0","M052",
                                      10.0233,kFALSE, 4.40000E-05,
                                     0.0,"Meson", 20553);
   new TAttParticle("Delta(1910)-","B012",
                                      1.91,kFALSE, .25,
                                     -1.,"Baryon", 21112);
   new TAttParticle("Delta(1910)+ bar","B012",
                                      1.91,kFALSE, .25,
                                     +1.,"Baryon", -21112);
   new TAttParticle("Delta(1920)-","B117",
                                      1.92,kFALSE, .2,
                                     -1.,"Baryon", 21114);
   new TAttParticle("Delta(1920)+ bar","B117",
                                      1.92,kFALSE, .2,
                                     +1.,"Baryon", -21114);
   new TAttParticle("Delta(1910)0","B012",
                                      1.91,kFALSE, .25,
                                     0.0,"Baryon", 21212);
   new TAttParticle("Delta(1910)0 bar","B012",
                                      1.91,kFALSE, .25,
                                     0.0,"Baryon", -21212);
   new TAttParticle("N(1700)0","B018",
                                      1.7,kFALSE, .1,
                                     0.0,"Baryon", 21214);
   new TAttParticle("N(1700)0 bar","B018",
                                      1.7,kFALSE, .1,
                                     0.0,"Baryon", -21214);
   new TAttParticle("N(1535)0","B063",
                                      1.535,kFALSE, .15,
                                     0.0,"Baryon", 22112);
   new TAttParticle("N(1535)0 bar","B063",
                                      1.535,kFALSE, .15,
                                     0.0,"Baryon", -22112);
   new TAttParticle("Delta(1920)0","B117",
                                      1.92,kFALSE, .2,
                                     0.0,"Baryon", 22114);
   new TAttParticle("Delta(1920)0 bar","B117",
                                      1.92,kFALSE, .2,
                                     0.0,"Baryon", -22114);
   new TAttParticle("Delta(1910)+","B012",
                                      1.91,kFALSE, .25,
                                     +1.,"Baryon", 22122);
   new TAttParticle("Delta(1910)- bar","B012",
                                      1.91,kFALSE, .25,
                                     -1.,"Baryon", -22122);
   new TAttParticle("N(1700)+","B018",
                                      1.7,kFALSE, .1,
                                     +1.,"Baryon", 22124);
   new TAttParticle("N(1700)- bar","B018",
                                      1.7,kFALSE, .1,
                                     -1.,"Baryon", -22124);
   new TAttParticle("N(1535)+","B063",
                                      1.535,kFALSE, .15,
                                     +1.,"Baryon", 22212);
   new TAttParticle("N(1535)- bar","B063",
                                      1.535,kFALSE, .15,
                                     -1.,"Baryon", -22212);
   new TAttParticle("Delta(1920)+","B117",
                                      1.92,kFALSE, .2,
                                     +1.,"Baryon", 22214);
   new TAttParticle("Delta(1920)- bar","B117",
                                      1.92,kFALSE, .2,
                                     -1.,"Baryon", -22214);
   new TAttParticle("Delta(1910)++","B012",
                                      1.91,kFALSE, .25,
                                     +2.,"Baryon", 22222);
   new TAttParticle("Delta(1910)-- bar","B012",
                                      1.91,kFALSE, .25,
                                     -2.,"Baryon", -22222);
   new TAttParticle("Delta(1920)++","B117",
                                      1.92,kFALSE, .2,
                                     +2.,"Baryon", 22224);
   new TAttParticle("Delta(1920)-- bar","B117",
                                      1.92,kFALSE, .2,
                                     -2.,"Baryon", -22224);
   new TAttParticle("Sigma(1750)-","B057",
                                      1.75,kFALSE, 9.00000E-02,
                                     -1.,"Baryon", 23112);
   new TAttParticle("Sigma(1750)+ bar","B057",
                                      1.75,kFALSE, 9.00000E-02,
                                     +1.,"Baryon", -23112);
   new TAttParticle("Sigma(1940)-","B098",
                                      1.94,kFALSE, .2199999,
                                     -1.,"Baryon", 23114);
   new TAttParticle("Sigma(1940)+ bar","B098",
                                      1.94,kFALSE, .2199999,
                                     +1.,"Baryon", -23114);
   new TAttParticle("Lambda(1600)0","B101",
                                      1.6,kFALSE, .15,
                                     0.0,"Baryon", 23122);
   new TAttParticle("Lambda(1600)0 bar","B101",
                                      1.6,kFALSE, .15,
                                     0.0,"Baryon", -23122);
   new TAttParticle("Lambda(1890)0","B060",
                                      1.89,kFALSE, .1,
                                     0.0,"Baryon", 23124);
   new TAttParticle("Lambda(1890)0 bar","B060",
                                      1.89,kFALSE, .1,
                                     0.0,"Baryon", -23124);
   new TAttParticle("Lambda(2110)0","B035",
                                      2.11,kFALSE, .2,
                                     0.0,"Baryon", 23126);
   new TAttParticle("Lambda(2110)0 bar","B035",
                                      2.11,kFALSE, .2,
                                     0.0,"Baryon", -23126);
   new TAttParticle("Sigma(1750)0","B057",
                                      1.75,kFALSE, 9.00000E-02,
                                     0.0,"Baryon", 23212);
   new TAttParticle("Sigma(1750)0 bar","B057",
                                      1.75,kFALSE, 9.00000E-02,
                                     0.0,"Baryon", -23212);
   new TAttParticle("Sigma(1940)0","B098",
                                      1.94,kFALSE, .2199999,
                                     0.0,"Baryon", 23214);
   new TAttParticle("Sigma(1940)0 bar","B098",
                                      1.94,kFALSE, .2199999,
                                     0.0,"Baryon", -23214);
   new TAttParticle("Sigma(1750)+","B057",
                                      1.75,kFALSE, 9.00000E-02,
                                     +1.,"Baryon", 23222);
   new TAttParticle("Sigma(1750)- bar","B057",
                                      1.75,kFALSE, 9.00000E-02,
                                     -1.,"Baryon", -23222);
   new TAttParticle("Sigma(1940)+","B098",
                                      1.94,kFALSE, .2199999,
                                     +1.,"Baryon", 23224);
   new TAttParticle("Sigma(1940)- bar","B098",
                                      1.94,kFALSE, .2199999,
                                     -1.,"Baryon", -23224);
   new TAttParticle("rho(1700)0","M065",
                                      1.7,kFALSE, .24,
                                     0.0,"Meson", 30113);
   new TAttParticle("rho(1700)+","M065",
                                      1.7,kFALSE, .24,
                                     1.,"Meson", 30213);
   new TAttParticle("rho(1700)-","M065",
                                      1.7,kFALSE, .24,
                                     -1.,"Meson", -30213);
   new TAttParticle("f(1)(1420)0","M006",
                                      1.4268,kFALSE, 5.20000E-02,
                                     0.0,"Meson", 30223);
   new TAttParticle("f(2)(2300)0","M107",
                                      2.297,kFALSE, .15,
                                     0.0,"Meson", 30225);
   new TAttParticle("K*(1410)0","M094",
                                      1.412,kFALSE, .2269999,
                                     0.0,"Meson", 30313);
   new TAttParticle("K*(1410)0 bar","M094",
                                      1.412,kFALSE, .2269999,
                                     0.0,"Meson", -30313);
   new TAttParticle("K*(1410)+","M094",
                                      1.412,kFALSE, .2269999,
                                     1.,"Meson", 30323);
   new TAttParticle("K*(1410)-","M094",
                                      1.412,kFALSE, .2269999,
                                     -1.,"Meson", -30323);
   new TAttParticle("psi(3770)0","M053",
                                      3.7699,kFALSE, 2.36000E-02,
                                     0.0,"Meson", 30443);
   new TAttParticle("Upsilon(3S)0","M048",
                                      10.35529,kFALSE, 2.63000E-05,
                                     0.0,"Meson", 30553);
   new TAttParticle("Delta(1600)-","B019",
                                      1.6,kFALSE, .3499999,
                                     -1.,"Baryon", 31114);
   new TAttParticle("Delta(1600)+ bar","B019",
                                      1.6,kFALSE, .3499999,
                                     +1.,"Baryon", -31114);
   new TAttParticle("N(1720)0","B015",
                                      1.72,kFALSE, .15,
                                     0.0,"Baryon", 31214);
   new TAttParticle("N(1720)0 bar","B015",
                                      1.72,kFALSE, .15,
                                     0.0,"Baryon", -31214);
   new TAttParticle("N(1650)0","B066",
                                      1.65,kFALSE, .15,
                                     0.0,"Baryon", 32112);
   new TAttParticle("N(1650)0 bar","B066",
                                      1.65,kFALSE, .15,
                                     0.0,"Baryon", -32112);
   new TAttParticle("Delta(1600)0","B019",
                                      1.6,kFALSE, .3499999,
                                     0.0,"Baryon", 32114);
   new TAttParticle("Delta(1600)0 bar","B019",
                                      1.6,kFALSE, .3499999,
                                     0.0,"Baryon", -32114);
   new TAttParticle("N(1720)+","B015",
                                      1.72,kFALSE, .15,
                                     +1.,"Baryon", 32124);
   new TAttParticle("N(1720)- bar","B015",
                                      1.72,kFALSE, .15,
                                     -1.,"Baryon", -32124);
   new TAttParticle("N(1650)+","B066",
                                      1.65,kFALSE, .15,
                                     +1.,"Baryon", 32212);
   new TAttParticle("N(1650)- bar","B066",
                                      1.65,kFALSE, .15,
                                     -1.,"Baryon", -32212);
   new TAttParticle("Delta(1600)+","B019",
                                      1.6,kFALSE, .3499999,
                                     +1.,"Baryon", 32214);
   new TAttParticle("Delta(1600)- bar","B019",
                                      1.6,kFALSE, .3499999,
                                     -1.,"Baryon", -32214);
   new TAttParticle("Delta(1600)++","B019",
                                      1.6,kFALSE, .3499999,
                                     +2.,"Baryon", 32224);
   new TAttParticle("Delta(1600)-- bar","B019",
                                      1.6,kFALSE, .3499999,
                                     -2.,"Baryon", -32224);
   new TAttParticle("Lambda(1670)0","B040",
                                      1.67,kFALSE, 3.50000E-02,
                                     0.0,"Baryon", 33122);
   new TAttParticle("Lambda(1670)0 bar","B040",
                                      1.67,kFALSE, 3.50000E-02,
                                     0.0,"Baryon", -33122);
   new TAttParticle("rho(1450)0","M105",
                                      1.465,kFALSE, .31,
                                     0.0,"Meson", 40113);
   new TAttParticle("rho(1450)+","M105",
                                      1.465,kFALSE, .31,
                                     1.,"Meson", 40213);
   new TAttParticle("rho(1450)-","M105",
                                      1.465,kFALSE, .31,
                                     -1.,"Meson", -40213);
   new TAttParticle("eta(1440)0","M027",
                                      1.42,kFALSE, 6.00000E-02,
                                     0.0,"Meson", 40221);
   new TAttParticle("f(1)(1510)0","M084",
                                      1.512,kFALSE, 3.50000E-02,
                                     0.0,"Meson", 40223);
   new TAttParticle("f(2)(2340)0","M108",
                                      2.34,kFALSE, .3199999,
                                     0.0,"Meson", 40225);
   new TAttParticle("K*(1680)0","M095",
                                      1.714,kFALSE, .3199999,
                                     0.0,"Meson", 40313);
   new TAttParticle("K*(1680)0 bar","M095",
                                      1.714,kFALSE, .3199999,
                                     0.0,"Meson", -40313);
   new TAttParticle("K*(1680)+","M095",
                                      1.714,kFALSE, .3199999,
                                     1.,"Meson", 40323);
   new TAttParticle("K*(1680)-","M095",
                                      1.714,kFALSE, .3199999,
                                     -1.,"Meson", -40323);
   new TAttParticle("psi(4040)0","M072",
                                      4.04,kFALSE, 5.20000E-02,
                                     0.0,"Meson", 40443);
   new TAttParticle("Upsilon(4S)0","M047",
                                      10.57999,kFALSE, 2.38000E-02,
                                     0.0,"Meson", 40553);
   new TAttParticle("N(1710)0","B014",
                                      1.71,kFALSE, .1,
                                     0.0,"Baryon", 42112);
   new TAttParticle("N(1710)0 bar","B014",
                                      1.71,kFALSE, .1,
                                     0.0,"Baryon", -42112);
   new TAttParticle("N(1710)+","B014",
                                      1.71,kFALSE, .1,
                                     +1.,"Baryon", 42212);
   new TAttParticle("N(1710)- bar","B014",
                                      1.71,kFALSE, .1,
                                     -1.,"Baryon", -42212);
   new TAttParticle("Lambda(1800)0","B036",
                                      1.8,kFALSE, .3,
                                     0.0,"Baryon", 43122);
   new TAttParticle("Lambda(1800)0 bar","B036",
                                      1.8,kFALSE, .3,
                                     0.0,"Baryon", -43122);
   new TAttParticle("f(0)(1590)0","M096",
                                      1.581,kFALSE, .18,
                                     0.0,"Meson", 50221);
   new TAttParticle("omega(1420)0","M125",
                                      1.419,kFALSE, .17,
                                     0.0,"Meson", 50223);
   new TAttParticle("psi(4160)0","M025",
                                      4.159,kFALSE, 7.80000E-02,
                                     0.0,"Meson", 50443);
   new TAttParticle("Upsilon(10860)0","M092",
                                      10.86499,kFALSE, .1099999,
                                     0.0,"Meson", 50553);
   new TAttParticle("Lambda(1810)0","B077",
                                      1.81,kFALSE, .15,
                                     0.0,"Baryon", 53122);
   new TAttParticle("Lambda(1810)0 bar","B077",
                                      1.81,kFALSE, .15,
                                     0.0,"Baryon", -53122);
   new TAttParticle("f(J)(1710)0","M068",
                                      1.709,kFALSE, .14,
                                     0.0,"Meson", 60221);
   new TAttParticle("omega(1600)0","M126",
                                      1.662,kFALSE, .28,
                                     0.0,"Meson", 60223);
   new TAttParticle("psi(4415)0","M073",
                                      4.415,kFALSE, 4.30000E-02,
                                     0.0,"Meson", 60443);
   new TAttParticle("Upsilon(11020)0","M093",
                                      11.019,kFALSE, 7.90000E-02,
                                     0.0,"Meson", 60553);
   new TAttParticle("chi(b1)(2P)0","M080",
                                      10.2552,kTRUE, .0,
                                     0.0,"Meson", 70553);
// End of the entry point of the pdg table conversion
   new TAttParticle("Rootino","",
                                    0.0,kTRUE,
                                    1.e38,0.0,"Artificial",0);
}

//______________________________________________________________________________
TAttParticle* TAttParticle::GetParticle(const char *name)
{
//
//  Get a pointer to the particle object according to the name given
//
   TAttParticle *def = (TAttParticle *)fgList->FindObject(name);
   if (!def) {
      fgList->Error("GetParticle","No match for %s exists !",name);
   }
   return def;
}

//______________________________________________________________________________
TAttParticle* TAttParticle::GetParticle(Int_t mcnumber)
{
//
//  Get a pointer to the particle object according to the MC code number
//
   TIter next(fgList);
   TAttParticle *par;
   while ((par = (TAttParticle *)next())) {
      if (par->GetMCNumber() == mcnumber) return par;
   }
   fgList->Error("GetParticle","No match for %d exists !",mcnumber);
   return 0;
}

//______________________________________________________________________________
void TAttParticle::Print(Option_t *) const
{
//
//  Print the entire information of this kind of particle
//

   Printf("\nParticle: %-15s  ",
          this->GetName());
   if (!fPDGStable) {
      Printf("Mass: %8f     DecayWidth: %8f  Charge : %8f",
              fPDGMass, fPDGDecayWidth, fPDGCharge);
   }
   else {
      Printf("Mass: %8f     DecayWidth: Stable  Charge : %8f",
              fPDGMass, fPDGCharge);
   }
   Printf(" ");
}

//______________________________________________________________________________
Double_t TAttParticle::SampleMass() const
{
//
//  Samples a mass according to the Breit-Wigner resonance distribution
//
   if ( fPDGStable || fPDGDecayWidth == 0.0 )
      return fPDGMass;
   else {
      return (fPDGMass+
             0.5*fPDGDecayWidth*
             TMath::Tan((2.0*gRandom->Rndm()-1.0)*TMath::Pi()*0.5));
   }
}

//______________________________________________________________________________
Double_t TAttParticle::SampleMass(Double_t widthcut) const
{
//
//  Samples a mass in the interval:
//
//  fPDGMass-widthcut*fPDGDecayWidtht - fPDGMass+widthcut*fPDGDecayWidth
//
//  according to the Breit-Wigner resonance distribution
//
   if ( fPDGStable || fPDGDecayWidth == 0.0 )
      return fPDGMass;
   else {
      return (fPDGMass+
             0.5*fPDGDecayWidth*
             TMath::Tan((2.0*gRandom->Rndm(0)-1.0)*TMath::ATan(2.0*widthcut)));
   }
}
