// @(#)root/vmc:$Id$
// Author: Andreas Morsch 13/04/2002

/*************************************************************************
 * Copyright (C) 2006, Rene Brun and Fons Rademakers.                    *
 * Copyright (C) 2002, ALICE Experiment at CERN.                         *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPDGCode
#define ROOT_TPDGCode
//
// Enumeration of the constants for the PDG particle IDs.
//

typedef enum {kRootino=0,kDown=1,kDownBar=-1,kUp=2,kUpBar=-2,kStrange=3,
	  kStrangeBar=-3,kCharm=4,kCharmBar=-4,kBottom=5,
	  kBottomBar=-5,kTop=6,kTopBar=-6,kGluon=21,kPythia92=92,
	  kDd1=1103,kDd1Bar=-1103,kUd0=2101,kUd0Bar=-2101,kUd1=2103,
	  kUd1Bar=-2103,kUu1=2203,kUu1Bar=-2203,kSd0=3101,kSd0Bar=-3101,
	  kSd1=3103,kSd1Bar=-3103,kSu0=3201,kSu0Bar=-3201,kSu1=3203,
	  kSu1Bar=-3203,kSearches0=7,kElectron=11,kPositron=-11,kNuE=12,
	  kNuEBar=-12,kMuonMinus=13,kMuonPlus=-13,kNuMu=14,kNuMuBar=-14,
	  kTauMinus=15,kTauPlus=-15,kNuTau=16,kNuTauBar=-16,kGamma=22,
	  kZ0=23,kWPlus=24,kWMinus=-24,kPi0=111,kRho770_0=113,
	  kA2_1320_0=115,kRho3_1690_0=117,kK0Long=130,kPiPlus=211,
	  kPiMinus=-211,kRho770Plus=213,kRho770Minus=-213,
	  kA2_1320Plus=215,kProton=2212,kProtonBar=-2212,
	  kNeutron=2112,kNeutronBar=-2112,kK0Short=310,kK0=311,
	  kK0Bar=-311,kKPlus=321,kKMinus=-321,kLambda0=3122,
	  kLambda1520=3124,kLambda0Bar=-3122,kSigmaMinus=3112,kSigmaBarPlus=-3112,
	  kSigmaPlus=3222,kSigmaBarMinus=-3222,kSigma0=3212,
	  kSigma0Bar=-3212,kXiMinus=3312,kXiPlusBar=-3312,
          kOmegaMinus=3334,kOmegaPlusBar=-3334}
PDG_t;

/*
   "a(2)(1320)-",     -215
   "rho(3)(1690)+",    217
   "rho(3)(1690)-",   -217
   "eta0",             221
   "omega(782)0"       223
   "f(2)(1270)0"       225
   "omega(3)(1670)0",  227
   "f(4)(2050)0",      229
   "K*(892)0","M018",  313
   "K*(892)0 bar",    -313
   "K(2)*(1430)0",     315
   "K(2)*(1430)0 bar",-315
   "K(3)*(1780)0",     317
   "K(3)*(1780)0 bar",-317
   "K(4)*(2045)0",     319
   "K(4)*(2045)0 bar",-319
   "K*(892)+",         323
   "K*(892)-",        -323
   "K(2)*(1430)+","M022",
                                      1.4254,kFALSE, 9.84000E-02,
                                     0.0,"Meson", 325));
   "K(2)*(1430)-","M022",
                                      1.4254,kFALSE, 9.84000E-02,
                                     0.0,"Meson", -325));
   "K(3)*(1780)+","M060",
                                      1.77,kFALSE, .164,
                                     0.0,"Meson", 327));
   "K(3)*(1780)-","M060",
                                      1.77,kFALSE, .164,
                                     0.0,"Meson", -327));
   "K(4)*(2045)+","M035",
                                      2.045,kFALSE, .1979999,
                                     0.0,"Meson", 329));
   "K(4)*(2045)-","M035",
                                      2.045,kFALSE, .1979999,
                                     0.0,"Meson", -329));
   "eta'(958)0","M002",
                                      .9577699,kFALSE, 2.01000E-04,
                                     0.0,"Meson", 331));
   "phi(1020)0","M004",
                                      1.01941,kFALSE, 4.43000E-03,
                                     0.0,"Meson", 333));
   "f(2)'(1525)0","M013",
                                      1.525,kFALSE, 7.60000E-02,
                                     0.0,"Meson", 335));
   "phi(3)(1850)0","M054",
                                      1.854,kFALSE, 8.70000E-02,
                                     0.0,"Meson", 337));
   "D+","S031",
                                      1.8694,kFALSE, 6.23000E-13,
                                     0.0,"Meson", 411));
   "D-","S031",
                                      1.8694,kFALSE, 6.23000E-13,
                                     0.0,"Meson", -411));
   "D*(2010)+","M062",
                                      2.01,kTRUE, .0,
                                     0.0,"Meson", 413));
   "D*(2010)-","M062",
                                      2.01,kTRUE, .0,
                                     0.0,"Meson", -413));
   "D(2)*(2460)+","M150",
                                      2.456,kFALSE, 2.30000E-02,
                                     0.0,"Meson", 415));
   "D(2)*(2460)-","M150",
                                      2.456,kFALSE, 2.30000E-02,
                                     0.0,"Meson", -415));
   "D0","S032",
                                      1.8646,kFALSE, 1.58600E-12,
                                     0.0,"Meson", 421));
   "D0 bar","S032",
                                      1.8646,kFALSE, 1.58600E-12,
                                     0.0,"Meson", -421));
   "D*(2007)0","M061",
                                      2.0067,kTRUE, .0,
                                     0.0,"Meson", 423));
   "D*(2007)0 bar","M061",
                                      2.0067,kTRUE, .0,
                                     0.0,"Meson", -423));
   "D(2)*(2460)0","M119",
                                      2.4577,kFALSE, 2.10000E-02,
                                     0.0,"Meson", 425));
   "D(2)*(2460)0 bar","M119",
                                      2.4577,kFALSE, 2.10000E-02,
                                     0.0,"Meson", -425));
   "D(s)+","S034",
                                      1.9685,kFALSE, 1.41000E-12,
                                     0.0,"Meson", 431));
   "D(s)-","S034",
                                      1.9685,kFALSE, 1.41000E-12,
                                     0.0,"Meson", -431));
   "D(s)*+","S074",
                                      2.11,kTRUE, .0,
                                     0.0,"Meson", 433));
   "D(s)*-","S074",
                                      2.11,kTRUE, .0,
                                     0.0,"Meson", -433));
   "eta(c)(1S)0","M026",
                                      2.9788,kFALSE, 1.03000E-02,
                                     0.0,"Meson", 441));
   "J/psi(1S)0","M070",
                                      3.09688,kFALSE, 8.80000E-05,
                                     0.0,"Meson", 443));
   "chi(c2)(1P)0","M057",
                                      3.55617,kFALSE, 2.00000E-03,
                                     0.0,"Meson", 445));
   "B0","S049",
                                      5.279,kFALSE, 4.39000E-13,
                                     0.0,"Meson", 511));
   "B0 bar","S049",
                                      5.279,kFALSE, 4.39000E-13,
                                     0.0,"Meson", -511));
   "B*0","S085",
                                      5.3248,kTRUE, .0,
                                     0.0,"Meson", 513));
   "B*0 bar","S085",
                                      5.3248,kTRUE, .0,
                                     0.0,"Meson", -513));
   "B+","S049",
                                      5.2787,kFALSE, 4.28000E-13,
                                     0.0,"Meson", 521));
   "B-","S049",
                                      5.2787,kFALSE, 4.28000E-13,
                                     0.0,"Meson", -521));
   "B*+","S085",
                                      5.3248,kTRUE, .0,
                                     0.0,"Meson", 523));
   "B*-","S085",
                                      5.3248,kTRUE, .0,
                                     0.0,"Meson", -523));
   "B(s)0","S086",
                                      5.375,kFALSE, 4.90000E-13,
                                     0.0,"Meson", 531));
   "B(s)0 bar","S086",
                                      5.375,kFALSE, 4.90000E-13,
                                     0.0,"Meson", -531));
   "chi(b0)(1P)0","M076",
                                      9.8598,kTRUE, .0,
                                     0.0,"Meson", 551));
   "chi(b0)(1P)0 bar","M076",
                                      9.8598,kTRUE, .0,
                                     0.0,"Meson", -551));
   "Upsilon(1S)0","M049",
                                      9.46037,kFALSE, 5.25000E-05,
                                     0.0,"Meson", 553));
   "chi(b2)(1P)0","M078",
                                      9.9132,kTRUE, .0,
                                     0.0,"Meson", 555));
   "Delta(1620)-","B082",
                                      1.62,kFALSE, .15,
                                     -1.,"Baryon", 1112));
   "Delta(1620)+ bar","B082",
                                      1.62,kFALSE, .15,
                                     +1.,"Baryon", -1112));
   "Delta(1232)-","B033",
                                      1.232,kFALSE, .1199999,
                                     -1.,"Baryon", 1114));
   "Delta(1232)+ bar","B033",
                                      1.232,kFALSE, .1199999,
                                     +1.,"Baryon", -1114));
   "Delta(1905)-","B011",
                                      1.905,kFALSE, .3499999,
                                     -1.,"Baryon", 1116));
   "Delta(1905)+ bar","B011",
                                      1.905,kFALSE, .3499999,
                                     +1.,"Baryon", -1116));
   "Delta(1950)-","B083",
                                      1.95,kFALSE, .3,
                                     -1.,"Baryon", 1118));
   "Delta(1950)+ bar","B083",
                                      1.95,kFALSE, .3,
                                     +1.,"Baryon", -1118));
   "Delta(1620)0","B082",
                                      1.62,kFALSE, .15,
                                     0.0,"Baryon", 1212));
   "Delta(1620)0 bar","B082",
                                      1.62,kFALSE, .15,
                                     0.0,"Baryon", -1212));
   "N(1520)0","B062",
                                      1.52,kFALSE, .1199999,
                                     0.0,"Baryon", 1214));
   "N(1520)0 bar","B062",
                                      1.52,kFALSE, .1199999,
                                     0.0,"Baryon", -1214));
   "Delta(1905)0","B011",
                                      1.905,kFALSE, .3499999,
                                     0.0,"Baryon", 1216));
   "Delta(1905)0 bar","B011",
                                      1.905,kFALSE, .3499999,
                                     0.0,"Baryon", -1216));
   "N(2190)0","B071",
                                      2.19,kFALSE, .4499999,
                                     0.0,"Baryon", 1218));
   "N(2190)0 bar","B071",
                                      2.19,kFALSE, .4499999,
                                     0.0,"Baryon", -1218));
   "n","S017",
                                      .9395656,kFALSE, 7.42100E-28,
                                     0.0,"Baryon", 2112));
   "n bar","S017",
                                      .9395656,kFALSE, 7.42100E-28,
                                     0.0,"Baryon", -2112));
   "Delta(1232)0","B033",
                                      1.232,kFALSE, .1199999,
                                     0.0,"Baryon", 2114));
   "Delta(1232)0 bar","B033",
                                      1.232,kFALSE, .1199999,
                                     0.0,"Baryon", -2114));
   "N(1675)0","B064",
                                      1.675,kFALSE, .15,
                                     0.0,"Baryon", 2116));
   "N(1675)0 bar","B064",
                                      1.675,kFALSE, .15,
                                     0.0,"Baryon", -2116));
   "Delta(1950)0","B083",
                                      1.95,kFALSE, .3,
                                     0.0,"Baryon", 2118));
   "Delta(1950)0 bar","B083",
                                      1.95,kFALSE, .3,
                                     0.0,"Baryon", -2118));
   "Delta(1620)+","B082",
                                      1.62,kFALSE, .15,
                                     +1.,"Baryon", 2122));
   "Delta(1620)- bar","B082",
                                      1.62,kFALSE, .15,
                                     -1.,"Baryon", -2122));
   "N(1520)+","B062",
                                      1.52,kFALSE, .1199999,
                                     +1.,"Baryon", 2124));
   "N(1520)- bar","B062",
                                      1.52,kFALSE, .1199999,
                                     -1.,"Baryon", -2124));
   "Delta(1905)+","B011",
                                      1.905,kFALSE, .3499999,
                                     +1.,"Baryon", 2126));
   "Delta(1905)- bar","B011",
                                      1.905,kFALSE, .3499999,
                                     -1.,"Baryon", -2126));
   "N(2190)+","B071",
                                      2.19,kFALSE, .4499999,
                                     +1.,"Baryon", 2128));
   "N(2190)- bar","B071",
                                      2.19,kFALSE, .4499999,
                                     -1.,"Baryon", -2128));

   "Delta(1232)+","B033",
                                      1.232,kFALSE, .1199999,
                                     +1.,"Baryon", 2214));
   "Delta(1232)- bar","B033",
                                      1.232,kFALSE, .1199999,
                                     -1.,"Baryon", -2214));
   "N(1675)+","B064",
                                      1.675,kFALSE, .15,
                                     +1.,"Baryon", 2216));
   "N(1675)- bar","B064",
                                      1.675,kFALSE, .15,
                                     -1.,"Baryon", -2216));
   "Delta(1950)+","B083",
                                      1.95,kFALSE, .3,
                                     +1.,"Baryon", 2218));
   "Delta(1950)- bar","B083",
                                      1.95,kFALSE, .3,
                                     -1.,"Baryon", -2218));
   "Delta(1620)++","B082",
                                      1.62,kFALSE, .15,
                                     +2.,"Baryon", 2222));
   "Delta(1620)-- bar","B082",
                                      1.62,kFALSE, .15,
                                     -2.,"Baryon", -2222));
   "Delta(1232)++","B033",
                                      1.232,kFALSE, .1199999,
                                     +2.,"Baryon", 2224));
   "Delta(1232)-- bar","B033",
                                      1.232,kFALSE, .1199999,
                                     -2.,"Baryon", -2224));
   "Delta(1905)++","B011",
                                      1.905,kFALSE, .3499999,
                                     +2.,"Baryon", 2226));
   "Delta(1905)-- bar","B011",
                                      1.905,kFALSE, .3499999,
                                     -2.,"Baryon", -2226));
   "Delta(1950)++","B083",
                                      1.95,kFALSE, .3,
                                     +2.,"Baryon", 2228));
   "Delta(1950)-- bar","B083",
                                      1.95,kFALSE, .3,
                                     -2.,"Baryon", -2228));
   "Sigma(1385)-","B043",
                                      1.3872,kFALSE, 3.94000E-02,
                                     -1.,"Baryon", 3114));
   "Sigma(1385)+ bar","B043",
                                      1.3872,kFALSE, 3.94000E-02,
                                     +1.,"Baryon", -3114));
   "Sigma(1775)-","B045",
                                      1.775,kFALSE, .1199999,
                                     -1.,"Baryon", 3116));
   "Sigma(1775)+ bar","B045",
                                      1.775,kFALSE, .1199999,
                                     +1.,"Baryon", -3116));
   "Sigma(2030)-","B047",
                                      2.03,kFALSE, .18,
                                     -1.,"Baryon", 3118));
   "Sigma(2030)+ bar","B047",
                                      2.03,kFALSE, .18,
                                     +1.,"Baryon", -3118));
   "Lambda(1520)0","B038",
                                      1.5195,kFALSE, 1.56000E-02,
                                     0.0,"Baryon", 3124));
   "Lambda(1520)0 bar","B038",
                                      1.5195,kFALSE, 1.56000E-02,
                                     0.0,"Baryon", -3124));
   "Lambda(1820)0","B039",
                                      1.82,kFALSE, 8.00000E-02,
                                     0.0,"Baryon", 3126));
   "Lambda(1820)0 bar","B039",
                                      1.82,kFALSE, 8.00000E-02,
                                     0.0,"Baryon", -3126));
   "Lambda(2100)0","B041",
                                      2.1,kFALSE, .2,
                                     0.0,"Baryon", 3128));
   "Lambda(2100)0 bar","B041",
                                      2.1,kFALSE, .2,
                                     0.0,"Baryon", -3128));
   "Sigma(1385)0","B043",
                                      1.3837,kFALSE, 3.60000E-02,
                                     0.0,"Baryon", 3214));
   "Sigma(1385)0 bar","B043",
                                      1.3837,kFALSE, 3.60000E-02,
                                     0.0,"Baryon", -3214));
   "Sigma(1775)0","B045",
                                      1.775,kFALSE, .1199999,
                                     0.0,"Baryon", 3216));
   "Sigma(1775)0 bar","B045",
                                      1.775,kFALSE, .1199999,
                                     0.0,"Baryon", -3216));
   "Sigma(2030)0","B047",
                                      2.03,kFALSE, .18,
                                     0.0,"Baryon", 3218));
   "Sigma(2030)0 bar","B047",
                                      2.03,kFALSE, .18,
                                     0.0,"Baryon", -3218));
   "Sigma(1385)+","B043",
                                      1.3828,kFALSE, 3.58000E-02,
                                     +1.,"Baryon", 3224));
   "Sigma(1385)- bar","B043",
                                      1.3828,kFALSE, 3.58000E-02,
                                     -1.,"Baryon", -3224));
   "Sigma(1775)+","B045",
                                      1.775,kFALSE, .1199999,
                                     +1.,"Baryon", 3226));
   "Sigma(1775)- bar","B045",
                                      1.775,kFALSE, .1199999,
                                     -1.,"Baryon", -3226));
   "Sigma(2030)+","B047",
                                      2.03,kFALSE, .18,
                                     +1.,"Baryon", 3228));
   "Sigma(2030)- bar","B047",
                                      2.03,kFALSE, .18,
                                     -1.,"Baryon", -3228));
   "Xi-","S022",
                                      1.32132,kFALSE, 4.02000E-15,
                                     -1.,"Baryon", 3312));
   "Xi+ bar","S022",
                                      1.32132,kFALSE, 4.02000E-15,
                                     +1.,"Baryon", -3312));
   "Xi(1530)-","B049",
                                      1.535,kFALSE, 9.90000E-03,
                                     -1.,"Baryon", 3314));
   "Xi(1530)+ bar","B049",
                                      1.535,kFALSE, 9.90000E-03,
                                     +1.,"Baryon", -3314));
   "Xi0","S023",
                                      1.3149,kFALSE, 2.27000E-15,
                                     0.0,"Baryon", 3322));
   "Xi0 bar","S023",
                                      1.3149,kFALSE, 2.27000E-15,
                                     0.0,"Baryon", -3322));
   "Xi(1530)0","B049",
                                      1.5318,kFALSE, 9.10000E-03,
                                     0.0,"Baryon", 3324));
   "Xi(1530)0 bar","B049",
                                      1.5318,kFALSE, 9.10000E-03,
                                     0.0,"Baryon", -3324));
   "Omega-","S024",
                                      1.67245,kFALSE, 8.01000E-15,
                                     -1.,"Baryon", 3334));
   "Omega+ bar","S024",
                                      1.67245,kFALSE, 8.01000E-15,
                                     +1.,"Baryon", -3334));
   "Sigma(c)(2455)0","B104",
                                      2.4524,kTRUE, .0,
                                     0.0,"Baryon", 4112));
   "Sigma(c)(2455)0 bar","B104",
                                      2.4524,kTRUE, .0,
                                     0.0,"Baryon", -4112));
   "Sigma(c)*0","Sigma(c)*0",
                                      -1.,kTRUE, -1,
                                     0.,"Baryon", 4114));
   "Sigma(c)*0 bar","Sigma(c)*0 bar",
                                      -1.,kTRUE, -1,
                                     0.,"Baryon", -4114));
   "Lambda(c)+","S033",
                                      2.2851,kFALSE, 3.29000E-12,
                                     +1.,"Baryon", 4122));
   "Lambda(c)- bar","S033",
                                      2.2851,kFALSE, 3.29000E-12,
                                     -1.,"Baryon", -4122));
   "Sigma(c)(2455)+","B104",
                                      2.4538,kTRUE, .0,
                                     +1.,"Baryon", 4212));
   "Sigma(c)(2455)- bar","B104",
                                      2.4538,kTRUE, .0,
                                     -1.,"Baryon", -4212));
   "Sigma(c)(2455)++","B104",
                                      2.4531,kTRUE, .0,
                                     +2.,"Baryon", 4222));
   "Sigma(c)(2455)-- bar","B104",
                                      2.4531,kTRUE, .0,
                                     -2.,"Baryon", -4222));
   "Sigma(c)*++","Sigma(c)*++",
                                      -1.,kTRUE, -1.,
                                     +2.,"Baryon", 4224));
   "Sigma(c)*++ bar","Sigma(c)*++ bar",
                                      -1.,kTRUE, -1.,
                                     -2.,"Baryon", -4224));
   "Xi(c)0","S048",
                                      2.4703,kFALSE, 6.70000E-12,
                                     0.0,"Baryon", 4312));
   "Xi(c)0 bar","S048",
                                      2.4703,kFALSE, 6.70000E-12,
                                     0.0,"Baryon", -4312));
   "Xi(c)+","S045",
                                      2.4651,kFALSE, 1.86000E-12,
                                     +1.,"Baryon", 4322));
   "Xi(c)- bar","S045",
                                      2.4651,kFALSE, 1.86000E-12,
                                     -1.,"Baryon", -4322));
//-----------------------------------------------------------------------------
// B-baryons
//-----------------------------------------------------------------------------
   "Lambda(b)0","S040",
                                      5.64,kFALSE, 6.20000E-13,
                                     0.0,"Baryon", 5122));
   "Lambda(b)0 bar","S040",
                                      5.64,kFALSE, 6.20000E-13,
                                     0.0,"Baryon", -5122));
   "Sigma(b)-","Sigma(b)-",
                                      -1.,kFALSE, -1.,
                                     0.0,"Baryon", 5112));
   "Sigma(b)- bar","Sigma(b)- bar",
                                      -1.,kFALSE, -1.,
                                     0.0,"Baryon", -5112));
   "Sigma(b)+","Sigma(b)+",
                                      -1.,kFALSE, -1.,
                                     0.0,"Baryon", 5222));
   "Sigma(b)+ bar","Sigma(b)+ bar",
                                      -1.,kFALSE, -1.,
                                     0.0,"Baryon", -5222));
   "Sigma(b)0","Sigma(b)0",
                                      -1.,kFALSE, -1.,
                                     0.0,"Baryon", 5212));
   "Sigma(b)0 bar","Sigma(b)0 bar",
                                      -1.,kFALSE, -1.,
                                     0.0,"Baryon", -5212));
   "Sigma(b)*-","Sigma(b)*-",
                                      -1.,kFALSE, -1.,
                                     0.0,"Baryon", 5114));
   "Sigma(b)*- bar","Sigma(b)*- bar",
                                      -1.,kFALSE, -1.,
                                     0.0,"Baryon", -5114));
   "Sigma(b)*+","Sigma(b)*+",
				       -1.,kFALSE, -1.,
				       1.0,"Baryon", 5214));
   "Sigma(b)*+ bar","Sigma(b)*+ bar",
				       -1.,kFALSE, -1.,
				       -1.0,"Baryon", -5214));
   "Ksi(b)-","Ksi(b)-",
                                      -1.,kFALSE, -1.,
                                     -1.0,"Baryon", 5132));
   "Ksi(b)- bar","Ksi(b)- bar",
                                      -1.,kFALSE, -1.,
                                      1.0,"Baryon", -5132));
//-----------------------------------------------------------------------------
   "a(0)(980)0","M036",
                                      .9824,kTRUE, .0,
                                     0.0,"Meson", 10111));
   "b(1)(1235)0","M011",
                                      1.231,kFALSE, .142,
                                     0.0,"Meson", 10113));
   "pi(2)(1670)0","M034",
                                      1.67,kFALSE, .2399999,
                                     0.0,"Meson", 10115));
   "a(0)(980)+","M036",
                                      .9824,kTRUE, .0,
                                     0.0,"Meson", 10211));
   "a(0)(980)-","M036",
                                      .9824,kTRUE, .0,
                                     0.0,"Meson", -10211));
   "b(1)(1235)+","M011",
                                      1.231,kFALSE, .142,
                                     0.0,"Meson", 10213));
   "b(1)(1235)-","M011",
                                      1.231,kFALSE, .142,
                                     0.0,"Meson", -10213));
   "pi(2)(1670)+","M034",
                                      1.67,kFALSE, .2399999,
                                     0.0,"Meson", 10215));
   "pi(2)(1670)-","M034",
                                      1.67,kFALSE, .2399999,
                                     0.0,"Meson", -10215));
   "f(0)(980)0","M003",
                                      .98,kTRUE, .0,
                                     0.0,"Meson", 10221));
   "h(1)(1170)0","M030",
                                      1.17,kFALSE, .36,
                                     0.0,"Meson", 10223));
   "K(0)*(1430)0","M019",
                                      1.429,kFALSE, .287,
                                     0.0,"Meson", 10311));
   "K(0)*(1430)0 bar","M019",
                                      1.429,kFALSE, .287,
                                     0.0,"Meson", -10311));
   "K(1)(1270)0","M028",
                                      1.272,kFALSE, 9.00000E-02,
                                     0.0,"Meson", 10313));
   "K(1)(1270)0 bar","M028",
                                      1.272,kFALSE, 9.00000E-02,
                                     0.0,"Meson", -10313));
   "K(2)(1770)0","M023",
                                      1.773,kFALSE, .186,
                                     0.0,"Meson", 10315));
   "K(2)(1770)0 bar","M023",
                                      1.773,kFALSE, .186,
                                     0.0,"Meson", -10315));
   "K(0)*(1430)+","M019",
                                      1.429,kFALSE, .287,
                                     0.0,"Meson", 10321));
   "K(0)*(1430)-","M019",
                                      1.429,kFALSE, .287,
                                     0.0,"Meson", -10321));
   "K(1)(1270)+","M028",
                                      1.272,kFALSE, 9.00000E-02,
                                     0.0,"Meson", 10323));
   "K(1)(1270)-","M028",
                                      1.272,kFALSE, 9.00000E-02,
                                     0.0,"Meson", -10323));
   "K(2)(1770)+","M023",
                                      1.773,kFALSE, .186,
                                     0.0,"Meson", 10325));
   "K(2)(1770)-","M023",
                                      1.773,kFALSE, .186,
                                     0.0,"Meson", -10325));
   "phi(1680)0","M067",
                                      1.68,kFALSE, .15,
                                     0.0,"Meson", 10333));
   "D(1)(2420)0","M097",
                                      2.4228,kFALSE, 1.80000E-02,
                                     0.0,"Meson", 10423));
   "D(s1)(2536)+","M121",
                                      2.53535,kTRUE, .0,
                                     0.0,"Meson", 10433));
   "D(s1)(2536)-","M121",
                                      2.53535,kTRUE, .0,
                                     0.0,"Meson", -10433));
   "chi(c0)(1P)0","M056",
                                      3.4151,kFALSE, 1.40000E-02,
                                     0.0,"Meson", 10441));
   "chi(c1)(1P)0","M055",
                                      3.51053,kFALSE, 8.80000E-04,
                                     0.0,"Meson", 10443));
   "chi(b0)(2P)0","M079",
                                      10.23209,kTRUE, .0,
                                     0.0,"Meson", 10551));
   "chi(b1)(1P)0","M077",
                                      9.8919,kTRUE, .0,
                                     0.0,"Meson", 10553));
   "chi(b2)(2P)0","M081",
                                      10.2685,kTRUE, .0,
                                     0.0,"Meson", 10555));
   "Delta(1900)-","B030",
                                      1.9,kFALSE, .2,
                                     -1.,"Baryon", 11112));
   "Delta(1900)+ bar","B030",
                                      1.9,kFALSE, .2,
                                     +1.,"Baryon", -11112));
   "Delta(1700)-","B010",
                                      1.7,kFALSE, .3,
                                     -1.,"Baryon", 11114));
   "Delta(1700)+ bar","B010",
                                      1.7,kFALSE, .3,
                                     +1.,"Baryon", -11114));
   "Delta(1930)-","B013",
                                      1.93,kFALSE, .3499999,
                                     -1.,"Baryon", 11116));
   "Delta(1930)+ bar","B013",
                                      1.93,kFALSE, .3499999,
                                     +1.,"Baryon", -11116));
   "Delta(1900)0","B030",
                                      1.9,kFALSE, .2,
                                     0.0,"Baryon", 11212));
   "Delta(1900)0 bar","B030",
                                      1.9,kFALSE, .2,
                                     0.0,"Baryon", -11212));
   "Delta(1930)0","B013",
                                      1.93,kFALSE, .3499999,
                                     0.0,"Baryon", 11216));
   "Delta(1930)0 bar","B013",
                                      1.93,kFALSE, .3499999,
                                     0.0,"Baryon", -11216));
   "N(1440)0","B061",
                                      1.44,kFALSE, .3499999,
                                     0.0,"Baryon", 12112));
   "N(1440)0 bar","B061",
                                      1.44,kFALSE, .3499999,
                                     0.0,"Baryon", -12112));
   "Delta(1700)0","B010",
                                      1.7,kFALSE, .3,
                                     0.0,"Baryon", 12114));
   "Delta(1700)0 bar","B010",
                                      1.7,kFALSE, .3,
                                     0.0,"Baryon", -12114));
   "N(1680)0","B065",
                                      1.68,kFALSE, .1299999,
                                     0.0,"Baryon", 12116));
   "N(1680)0 bar","B065",
                                      1.68,kFALSE, .1299999,
                                     0.0,"Baryon", -12116));
   "Delta(1900)+","B030",
                                      1.9,kFALSE, .2,
                                     +1.,"Baryon", 12122));
   "Delta(1900)- bar","B030",
                                      1.9,kFALSE, .2,
                                     -1.,"Baryon", -12122));
   "Delta(1930)+","B013",
                                      1.93,kFALSE, .3499999,
                                     +1.,"Baryon", 12126));
   "Delta(1930)- bar","B013",
                                      1.93,kFALSE, .3499999,
                                     -1.,"Baryon", -12126));
   "N(1440)+","B061",
                                      1.44,kFALSE, .3499999,
                                     +1.,"Baryon", 12212));
   "N(1440)- bar","B061",
                                      1.44,kFALSE, .3499999,
                                     -1.,"Baryon", -12212));
   "Delta(1700)+","B010",
                                      1.7,kFALSE, .3,
                                     +1.,"Baryon", 12214));
   "Delta(1700)- bar","B010",
                                      1.7,kFALSE, .3,
                                     -1.,"Baryon", -12214));
   "N(1680)+","B065",
                                      1.68,kFALSE, .1299999,
                                     +1.,"Baryon", 12216));
   "N(1680)- bar","B065",
                                      1.68,kFALSE, .1299999,
                                     -1.,"Baryon", -12216));
   "Delta(1900)++","B030",
                                      1.9,kFALSE, .2,
                                     +2.,"Baryon", 12222));
   "Delta(1900)-- bar","B030",
                                      1.9,kFALSE, .2,
                                     -2.,"Baryon", -12222));
   "Delta(1700)++","B010",
                                      1.7,kFALSE, .3,
                                     +2.,"Baryon", 12224));
   "Delta(1700)-- bar","B010",
                                      1.7,kFALSE, .3,
                                     -2.,"Baryon", -12224));
   "Delta(1930)++","B013",
                                      1.93,kFALSE, .3499999,
                                     +2.,"Baryon", 12226));
   "Delta(1930)-- bar","B013",
                                      1.93,kFALSE, .3499999,
                                     -2.,"Baryon", -12226));
   "Sigma(1660)-","B079",
                                      1.66,kFALSE, .1,
                                     -1.,"Baryon", 13112));
   "Sigma(1660)+ bar","B079",
                                      1.66,kFALSE, .1,
                                     +1.,"Baryon", -13112));
   "Sigma(1670)-","B051",
                                      1.67,kFALSE, 6.00000E-02,
                                     -1.,"Baryon", 13114));
   "Sigma(1670)+ bar","B051",
                                      1.67,kFALSE, 6.00000E-02,
                                     +1.,"Baryon", -13114));
   "Sigma(1915)-","B046",
                                      1.915,kFALSE, .1199999,
                                     -1.,"Baryon", 13116));
   "Sigma(1915)+ bar","B046",
                                      1.915,kFALSE, .1199999,
                                     +1.,"Baryon", -13116));
   "Lambda(1405)0","B037",
                                      1.407,kFALSE, 5.00000E-02,
                                     0.0,"Baryon", 13122));
   "Lambda(1405)0 bar","B037",
                                      1.407,kFALSE, 5.00000E-02,
                                     0.0,"Baryon", -13122));
   "Lambda(1690)0","B055",
                                      1.69,kFALSE, 6.00000E-02,
                                     0.0,"Baryon", 13124));
   "Lambda(1690)0 bar","B055",
                                      1.69,kFALSE, 6.00000E-02,
                                     0.0,"Baryon", -13124));
   "Lambda(1830)0","B056",
                                      1.83,kFALSE, 9.50000E-02,
                                     0.0,"Baryon", 13126));
   "Lambda(1830)0 bar","B056",
                                      1.83,kFALSE, 9.50000E-02,
                                     0.0,"Baryon", -13126));
   "Sigma(1660)0","B079",
                                      1.66,kFALSE, .1,
                                     0.0,"Baryon", 13212));
   "Sigma(1660)0 bar","B079",
                                      1.66,kFALSE, .1,
                                     0.0,"Baryon", -13212));
   "Sigma(1670)0","B051",
                                      1.67,kFALSE, 6.00000E-02,
                                     0.0,"Baryon", 13214));
   "Sigma(1670)0 bar","B051",
                                      1.67,kFALSE, 6.00000E-02,
                                     0.0,"Baryon", -13214));
   "Sigma(1915)0","B046",
                                      1.915,kFALSE, .1199999,
                                     0.0,"Baryon", 13216));
   "Sigma(1915)0 bar","B046",
                                      1.915,kFALSE, .1199999,
                                     0.0,"Baryon", -13216));
   "Sigma(1660)+","B079",
                                      1.66,kFALSE, .1,
                                     +1.,"Baryon", 13222));
   "Sigma(1660)- bar","B079",
                                      1.66,kFALSE, .1,
                                     -1.,"Baryon", -13222));
   "Sigma(1670)+","B051",
                                      1.67,kFALSE, 6.00000E-02,
                                     +1.,"Baryon", 13224));
   "Sigma(1670)- bar","B051",
                                      1.67,kFALSE, 6.00000E-02,
                                     -1.,"Baryon", -13224));
   "Sigma(1915)+","B046",
                                      1.915,kFALSE, .1199999,
                                     +1.,"Baryon", 13226));
   "Sigma(1915)- bar","B046",
                                      1.915,kFALSE, .1199999,
                                     -1.,"Baryon", -13226));
   "Xi(1820)-","B050",
                                      1.823,kFALSE, 2.40000E-02,
                                     -1.,"Baryon", 13314));
   "Xi(1820)+ bar","B050",
                                      1.823,kFALSE, 2.40000E-02,
                                     +1.,"Baryon", -13314));
   "Xi(1820)0","B050",
                                      1.823,kFALSE, 2.40000E-02,
                                     0.0,"Baryon", 13324));
   "Xi(1820)0 bar","B050",
                                      1.823,kFALSE, 2.40000E-02,
                                     0.0,"Baryon", -13324));
   "pi(1300)0","M058",
                                      1.3,kTRUE, .0,
                                     0.0,"Meson", 20111));
   "a(1)(1260)0","M010",
                                      1.23,kTRUE, .0,
                                     0.0,"Meson", 20113));
   "pi(1300)+","M058",
                                      1.3,kTRUE, .0,
                                     0.0,"Meson", 20211));
   "pi(1300)-","M058",
                                      1.3,kTRUE, .0,
                                     0.0,"Meson", -20211));
   "a(1)(1260)+","M010",
                                      1.23,kTRUE, .0,
                                     0.0,"Meson", 20213));
   "a(1)(1260)-","M010",
                                      1.23,kTRUE, .0,
                                     0.0,"Meson", -20213));
   "eta(1295)0","M037",
                                      1.295,kFALSE, 5.30000E-02,
                                     0.0,"Meson", 20221));
   "f(1)(1285)0","M008",
                                      1.282,kFALSE, 2.40000E-02,
                                     0.0,"Meson", 20223));
   "f(2)(2010)0","M106",
                                      2.01,kFALSE, .2,
                                     0.0,"Meson", 20225));
   "K(1)(1400)0","M064",
                                      1.402,kFALSE, .1739999,
                                     0.0,"Meson", 20313));
   "K(1)(1400)0 bar","M064",
                                      1.402,kFALSE, .1739999,
                                     0.0,"Meson", -20313));
   "K(2)(1820)0","M146",
                                      1.816,kFALSE, .2759999,
                                     0.0,"Meson", 20315));
   "K(2)(1820)0 bar","M146",
                                      1.816,kFALSE, .2759999,
                                     0.0,"Meson", -20315));
   "K(1)(1400)+","M064",
                                      1.402,kFALSE, .1739999,
                                     0.0,"Meson", 20323));
   "K(1)(1400)-","M064",
                                      1.402,kFALSE, .1739999,
                                     0.0,"Meson", -20323));
   "K(2)(1820)+","M146",
                                      1.816,kFALSE, .2759999,
                                     0.0,"Meson", 20325));
   "K(2)(1820)-","M146",
                                      1.816,kFALSE, .2759999,
                                     0.0,"Meson", -20325));
   "psi(2S)0","M071",
                                      3.686,kFALSE, 2.77000E-04,
                                     0.0,"Meson", 20443));
   "Upsilon(2S)0","M052",
                                      10.0233,kFALSE, 4.40000E-05,
                                     0.0,"Meson", 20553));
   "Delta(1910)-","B012",
                                      1.91,kFALSE, .25,
                                     -1.,"Baryon", 21112));
   "Delta(1910)+ bar","B012",
                                      1.91,kFALSE, .25,
                                     +1.,"Baryon", -21112));
   "Delta(1920)-","B117",
                                      1.92,kFALSE, .2,
                                     -1.,"Baryon", 21114));
   "Delta(1920)+ bar","B117",
                                      1.92,kFALSE, .2,
                                     +1.,"Baryon", -21114));
   "Delta(1910)0","B012",
                                      1.91,kFALSE, .25,
                                     0.0,"Baryon", 21212));
   "Delta(1910)0 bar","B012",
                                      1.91,kFALSE, .25,
                                     0.0,"Baryon", -21212));
   "N(1700)0","B018",
                                      1.7,kFALSE, .1,
                                     0.0,"Baryon", 21214));
   "N(1700)0 bar","B018",
                                      1.7,kFALSE, .1,
                                     0.0,"Baryon", -21214));
   "N(1535)0","B063",
                                      1.535,kFALSE, .15,
                                     0.0,"Baryon", 22112));
   "N(1535)0 bar","B063",
                                      1.535,kFALSE, .15,
                                     0.0,"Baryon", -22112));
   "Delta(1920)0","B117",
                                      1.92,kFALSE, .2,
                                     0.0,"Baryon", 22114));
   "Delta(1920)0 bar","B117",
                                      1.92,kFALSE, .2,
                                     0.0,"Baryon", -22114));
   "Delta(1910)+","B012",
                                      1.91,kFALSE, .25,
                                     +1.,"Baryon", 22122));
   "Delta(1910)- bar","B012",
                                      1.91,kFALSE, .25,
                                     -1.,"Baryon", -22122));
   "N(1700)+","B018",
                                      1.7,kFALSE, .1,
                                     +1.,"Baryon", 22124));
   "N(1700)- bar","B018",
                                      1.7,kFALSE, .1,
                                     -1.,"Baryon", -22124));
   "N(1535)+","B063",
                                      1.535,kFALSE, .15,
                                     +1.,"Baryon", 22212));
   "N(1535)- bar","B063",
                                      1.535,kFALSE, .15,
                                     -1.,"Baryon", -22212));
   "Delta(1920)+","B117",
                                      1.92,kFALSE, .2,
                                     +1.,"Baryon", 22214));
   "Delta(1920)- bar","B117",
                                      1.92,kFALSE, .2,
                                     -1.,"Baryon", -22214));
   "Delta(1910)++","B012",
                                      1.91,kFALSE, .25,
                                     +2.,"Baryon", 22222));
   "Delta(1910)-- bar","B012",
                                      1.91,kFALSE, .25,
                                     -2.,"Baryon", -22222));
   "Delta(1920)++","B117",
                                      1.92,kFALSE, .2,
                                     +2.,"Baryon", 22224));
   "Delta(1920)-- bar","B117",
                                      1.92,kFALSE, .2,
                                     -2.,"Baryon", -22224));
   "Sigma(1750)-","B057",
                                      1.75,kFALSE, 9.00000E-02,
                                     -1.,"Baryon", 23112));
   "Sigma(1750)+ bar","B057",
                                      1.75,kFALSE, 9.00000E-02,
                                     +1.,"Baryon", -23112));
   "Sigma(1940)-","B098",
                                      1.94,kFALSE, .2199999,
                                     -1.,"Baryon", 23114));
   "Sigma(1940)+ bar","B098",
                                      1.94,kFALSE, .2199999,
                                     +1.,"Baryon", -23114));
   "Lambda(1600)0","B101",
                                      1.6,kFALSE, .15,
                                     0.0,"Baryon", 23122));
   "Lambda(1600)0 bar","B101",
                                      1.6,kFALSE, .15,
                                     0.0,"Baryon", -23122));
   "Lambda(1890)0","B060",
                                      1.89,kFALSE, .1,
                                     0.0,"Baryon", 23124));
   "Lambda(1890)0 bar","B060",
                                      1.89,kFALSE, .1,
                                     0.0,"Baryon", -23124));
   "Lambda(2110)0","B035",
                                      2.11,kFALSE, .2,
                                     0.0,"Baryon", 23126));
   "Lambda(2110)0 bar","B035",
                                      2.11,kFALSE, .2,
                                     0.0,"Baryon", -23126));
   "Sigma(1750)0","B057",
                                      1.75,kFALSE, 9.00000E-02,
                                     0.0,"Baryon", 23212));
   "Sigma(1750)0 bar","B057",
                                      1.75,kFALSE, 9.00000E-02,
                                     0.0,"Baryon", -23212));
   "Sigma(1940)0","B098",
                                      1.94,kFALSE, .2199999,
                                     0.0,"Baryon", 23214));
   "Sigma(1940)0 bar","B098",
                                      1.94,kFALSE, .2199999,
                                     0.0,"Baryon", -23214));
   "Sigma(1750)+","B057",
                                      1.75,kFALSE, 9.00000E-02,
                                     +1.,"Baryon", 23222));
   "Sigma(1750)- bar","B057",
                                      1.75,kFALSE, 9.00000E-02,
                                     -1.,"Baryon", -23222));
   "Sigma(1940)+","B098",
                                      1.94,kFALSE, .2199999,
                                     +1.,"Baryon", 23224));
   "Sigma(1940)- bar","B098",
                                      1.94,kFALSE, .2199999,
                                     -1.,"Baryon", -23224));
   "rho(1700)0","M065",
                                      1.7,kFALSE, .23,
                                     0.0,"Meson", 30113));
   "rho(1700)+","M065",
                                      1.7,kFALSE, .23,
                                     0.0,"Meson", 30213));
   "rho(1700)-","M065",
                                      1.7,kFALSE, .23,
                                     0.0,"Meson", -30213));
   "f(1)(1420)0","M006",
                                      1.4268,kFALSE, 5.20000E-02,
                                     0.0,"Meson", 30223));
   "f(2)(2300)0","M107",
                                      2.297,kFALSE, .15,
                                     0.0,"Meson", 30225));
   "K*(1410)0","M094",
                                      1.412,kFALSE, .2269999,
                                     0.0,"Meson", 30313));
   "K*(1410)0 bar","M094",
                                      1.412,kFALSE, .2269999,
                                     0.0,"Meson", -30313));
   "K*(1410)+","M094",
                                      1.412,kFALSE, .2269999,
                                     0.0,"Meson", 30323));
   "K*(1410)-","M094",
                                      1.412,kFALSE, .2269999,
                                     0.0,"Meson", -30323));
   "psi(3770)0","M053",
                                      3.7699,kFALSE, 2.36000E-02,
                                     0.0,"Meson", 30443));
   "Upsilon(3S)0","M048",
                                      10.35529,kFALSE, 2.63000E-05,
                                     0.0,"Meson", 30553));
   "Delta(1600)-","B019",
                                      1.6,kFALSE, .3499999,
                                     -1.,"Baryon", 31114));
   "Delta(1600)+ bar","B019",
                                      1.6,kFALSE, .3499999,
                                     +1.,"Baryon", -31114));
   "N(1720)0","B015",
                                      1.72,kFALSE, .15,
                                     0.0,"Baryon", 31214));
   "N(1720)0 bar","B015",
                                      1.72,kFALSE, .15,
                                     0.0,"Baryon", -31214));
   "N(1650)0","B066",
                                      1.65,kFALSE, .15,
                                     0.0,"Baryon", 32112));
   "N(1650)0 bar","B066",
                                      1.65,kFALSE, .15,
                                     0.0,"Baryon", -32112));
   "Delta(1600)0","B019",
                                      1.6,kFALSE, .3499999,
                                     0.0,"Baryon", 32114));
   "Delta(1600)0 bar","B019",
                                      1.6,kFALSE, .3499999,
                                     0.0,"Baryon", -32114));
   "N(1720)+","B015",
                                      1.72,kFALSE, .15,
                                     +1.,"Baryon", 32124));
   "N(1720)- bar","B015",
                                      1.72,kFALSE, .15,
                                     -1.,"Baryon", -32124));
   "N(1650)+","B066",
                                      1.65,kFALSE, .15,
                                     +1.,"Baryon", 32212));
   "N(1650)- bar","B066",
                                      1.65,kFALSE, .15,
                                     -1.,"Baryon", -32212));
   "Delta(1600)+","B019",
                                      1.6,kFALSE, .3499999,
                                     +1.,"Baryon", 32214));
   "Delta(1600)- bar","B019",
                                      1.6,kFALSE, .3499999,
                                     -1.,"Baryon", -32214));
   "Delta(1600)++","B019",
                                      1.6,kFALSE, .3499999,
                                     +2.,"Baryon", 32224));
   "Delta(1600)-- bar","B019",
                                      1.6,kFALSE, .3499999,
                                     -2.,"Baryon", -32224));
   "Lambda(1670)0","B040",
                                      1.67,kFALSE, 3.50000E-02,
                                     0.0,"Baryon", 33122));
   "Lambda(1670)0 bar","B040",
                                      1.67,kFALSE, 3.50000E-02,
                                     0.0,"Baryon", -33122));
   "rho(1450)0","M105",
                                      1.465,kFALSE, .31,
                                     0.0,"Meson", 40113));
   "rho(1450)+","M105",
                                      1.465,kFALSE, .31,
                                     0.0,"Meson", 40213));
   "rho(1450)-","M105",
                                      1.465,kFALSE, .31,
                                     0.0,"Meson", -40213));
   "eta(1440)0","M027",
                                      1.42,kFALSE, 6.00000E-02,
                                     0.0,"Meson", 40221));
   "f(1)(1510)0","M084",
                                      1.512,kFALSE, 3.50000E-02,
                                     0.0,"Meson", 40223));
   "f(2)(2340)0","M108",
                                      2.34,kFALSE, .3199999,
                                     0.0,"Meson", 40225));
   "K*(1680)0","M095",
                                      1.714,kFALSE, .3199999,
                                     0.0,"Meson", 40313));
   "K*(1680)0 bar","M095",
                                      1.714,kFALSE, .3199999,
                                     0.0,"Meson", -40313));
   "K*(1680)+","M095",
                                      1.714,kFALSE, .3199999,
                                     0.0,"Meson", 40323));
   "K*(1680)-","M095",
                                      1.714,kFALSE, .3199999,
                                     0.0,"Meson", -40323));
   "psi(4040)0","M072",
                                      4.04,kFALSE, 5.20000E-02,
                                     0.0,"Meson", 40443));
   "Upsilon(4S)0","M047",
                                      10.57999,kFALSE, 2.38000E-02,
                                     0.0,"Meson", 40553));
   "N(1710)0","B014",
                                      1.71,kFALSE, .1,
                                     0.0,"Baryon", 42112));
   "N(1710)0 bar","B014",
                                      1.71,kFALSE, .1,
                                     0.0,"Baryon", -42112));
   "N(1710)+","B014",
                                      1.71,kFALSE, .1,
                                     +1.,"Baryon", 42212));
   "N(1710)- bar","B014",
                                      1.71,kFALSE, .1,
                                     -1.,"Baryon", -42212));
   "Lambda(1800)0","B036",
                                      1.8,kFALSE, .3,
                                     0.0,"Baryon", 43122));
   "Lambda(1800)0 bar","B036",
                                      1.8,kFALSE, .3,
                                     0.0,"Baryon", -43122));
   "f(0)(1590)0","M096",
                                      1.581,kFALSE, .18,
                                     0.0,"Meson", 50221));
   "omega(1420)0","M125",
                                      1.419,kFALSE, .17,
                                     0.0,"Meson", 50223));
   "psi(4160)0","M025",
                                      4.159,kFALSE, 7.80000E-02,
                                     0.0,"Meson", 50443));
   "Upsilon(10860)0","M092",
                                      10.86499,kFALSE, .1099999,
                                     0.0,"Meson", 50553));
   "Lambda(1810)0","B077",
                                      1.81,kFALSE, .15,
                                     0.0,"Baryon", 53122));
   "Lambda(1810)0 bar","B077",
                                      1.81,kFALSE, .15,
                                     0.0,"Baryon", -53122));
   "f(J)(1710)0","M068",
                                      1.709,kFALSE, .14,
                                     0.0,"Meson", 60221));
   "omega(1600)0","M126",
                                      1.662,kFALSE, .28,
                                     0.0,"Meson", 60223));
   "psi(4415)0","M073",
                                      4.415,kFALSE, 4.30000E-02,
                                     0.0,"Meson", 60443));
   "Upsilon(11020)0","M093",
                                      11.019,kFALSE, 7.90000E-02,
                                     0.0,"Meson", 60553));
   "chi(b1)(2P)0","M080",
                                      10.2552,kTRUE, .0,
                                     0.0,"Meson", 70553));
	    */


#endif //ROOT_TPDGCode
