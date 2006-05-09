// @(#)root/tmva $Id: TMVA_MethodCFMlpANN_utils.cxx,v 1.1 2006/05/08 12:46:31 brun Exp $ 
// Author: Andreas Hoecker, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_MethodCFMlpANN_utils                                             *
 *                                                                                *
 * Reference for the original FORTRAN version "mlpl3.F":                          *
 *      Authors  : J. Proriol and contributions from ALEPH-Clermont-Fd            *
 *                 Team members                                                   *
 *      Copyright: Laboratoire Physique Corpusculaire                             *
 *                 Universite de Blaise Pascal, IN2P3/CNRS                        *
 *                                                                                *
 * Modifications by present authors:                                              *
 *      translation through f2c                                                   *
 *      removal of output-related functionality                                   *
 *      removal of f2c special types                                              *
 *      use dynamical data tables (not for all of them, but for the big ones)     *
 *                                                                                *
 * Description:                                                                   *
 *      Utility routine, obtained via f2c from original mlpl3.F FORTRAN routine   *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany     *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-KP Heidelberg, Germany,                                               * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 *                                                                                *
 * File and Version Information:                                                  *
 * $Id: TMVA_MethodCFMlpANN_utils.cxx,v 1.1 2006/05/08 12:46:31 brun Exp $ 
 **********************************************************************************/

#include "TMVA_MethodCFMlpANN_utils.h"
#include "TMVA_MethodCFMlpANN_def.h"
#include "TMVA_Timer.h"
#include <stdio.h>
#include "string.h"

#ifndef R__WIN32
//_______________________________________________________________________
//                                                                      
// CFMlpANN implementation 
//                                                                      
//_______________________________________________________________________


#ifdef KR_headers
extern void f_exit();
void s_stop(s, n) char *s; Int_t n;
#else
#undef abs
#undef min
#undef max
#include <stdlib.h>
#ifdef __cplusplus
extern "C" {
#endif
void f_exit(void);

Int_t s_stop(char *s, Int_t n)
#endif
{
Int_t i;

if(n > 0)
	{
	fprintf(stderr, "STOP ");
	for(i = 0; i<n ; ++i)
		putc(*s++, stderr);
	fprintf(stderr, " statement executed\n");
	}
#ifdef NO_ONEXIT
f_exit();
#endif
exit(0);

// We cannot avoid (useless) compiler diagnostics here:		
// some compilers complain if there is no return statement,	
// and others complain that this one cannot be reached.	

return 0; /* NOT REACHED */
}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
  
#define datacc_          TMVA_MethodCFMlpANN_dataInterface
#define writeWeightFile_ TMVA_MethodCFMlpANN_writeWeightsToFile
  
  /* Common Block Declarations */

  struct {
    Int_t nnfillntuple;
  } nnfillntuplecom_;

#define nnfillntuplecom_1 nnfillntuplecom_

  struct {
    Double_t epsmin, epsmax, eeps, eta;
    Int_t layerm, lclass, nevl, nblearn, nunilec, nunisor, nunishort, nunap,
      nvar, itest, ndiv, ichoi, ndivis, nevt;
  } param_;

#define param_1 param_

  struct {
    Double_t xmax[max_nVar_], xmin[max_nVar_];
    Int_t nclass[max_Events_], mclass[max_Events_], iclass;
  } varn_;

#define varn_1 varn_

  struct {
    // AH: introduced dynamical data table; original array was
    //     (* Double_t xeev[max_Events_*max_nVar_]; // was [max_Events_][max_nVar_] *)
    void create( Int_t nevt, Int_t nvar ) {
      fNevt = nevt+1; fNvar = nvar+1; // fortran array style 1...N
      xeev = new Double_t*[fNevt];
      for (Int_t i=0; i<fNevt; i++) xeev[i] = new Double_t[fNvar];
    }
    Double_t operator=( Double_t val ) { return val; }
    Double_t &operator()( Int_t ievt, Int_t ivar ) const { 
      if (0 != xeev && ievt < fNevt && ivar < fNvar) return xeev[ievt][ivar];
      else {
	printf( "*** ERROR in varn2_(): xeev is zero pointer ==> abort ***\n") ;
	s_stop("", (Int_t)0);
      }
    }
    void Delete( void ) {
      if (0 != xeev) for (Int_t i=0; i<fNevt; i++) if (0 != xeev[i]) delete [] xeev[i];
    }

    Double_t** xeev;
    Int_t fNevt;
    Int_t fNvar;
  } varn2_;

#define varn2_1 varn2_

  struct {
    // AH: introduced dynamical data table; original array was
    //     (* Double_t xx[max_Events_*max_nVar_]; // was [max_Events_][max_nVar_] *)
    void create( Int_t nevt, Int_t nvar ) {
      fNevt = nevt+1; fNvar = nvar+1; // fortran array style 1...N
      xx = new Double_t*[fNevt];
      for (Int_t i=0; i<fNevt; i++) xx[i] = new Double_t[fNvar];
    }
    Double_t operator=( Double_t val ) { return val; }
    Double_t &operator()( Int_t ievt, Int_t ivar ) const { 
      if (0 != xx && ievt < fNevt && ivar < fNvar) return xx[ievt][ivar];
      else {
	printf( "*** ERROR in varn3_(): xx is zero pointer ==> abort ***\n") ;
	s_stop("", (Int_t)0);
      }
    }
    void Delete( void ) {
      if (0 != xx) for (Int_t i=0; i<fNevt; i++) if (0 != xx[i]) delete [] xx[i];
    }

    Double_t** xx;
    Int_t fNevt;
    Int_t fNvar;
  } varn3_;

#define varn3_1 varn3_

  struct {
    Double_t x[max_nLayers_*max_nNodes_];	      // was [max_nLayers_][max_nNodes_] 
    Double_t y[max_nLayers_*max_nNodes_];	      // was [max_nLayers_][max_nNodes_] 
    Double_t o[max_nNodes_];
    Double_t w[max_nLayers_*max_nNodes_*max_nNodes_]; // was [max_nLayers_][max_nNodes_][max_nNodes_] 
    Double_t ww[max_nLayers_*max_nNodes_];            // was [max_nLayers_][max_nNodes_] 
    Double_t cut[max_nNodes_], deltaww[max_nLayers_*max_nNodes_];  // was [max_nLayers_][max_nNodes_] 
    Int_t neuron[max_nLayers_];
  } neur_;

#define neur_1 neur_

  struct {
    Double_t coef[max_nNodes_], temp[max_nLayers_], demin, demax;
    Double_t del[max_nLayers_*max_nNodes_];	          // was [max_nLayers_][max_nNodes_]
    Double_t delw[max_nLayers_*max_nNodes_*max_nNodes_];  // was [max_nLayers_][max_nNodes_][max_nNodes_]
    Double_t delta[max_nLayers_*max_nNodes_*max_nNodes_]; // was [max_nLayers_][max_nNodes_][max_nNodes_]
    Double_t delww[max_nLayers_*max_nNodes_];	          // was [max_nLayers_][max_nNodes_]
    Int_t idde;
  } del_;

#define del_1 del_

  struct {
    Double_t ancout, tolcou;
    Int_t ieps;
  } cost_;

#define cost_1 cost_

  struct {
    Int_t multinn;
  } multinn_;

#define multinn_1 multinn_

  /* Table of constant values */

  static Int_t c__100 = 100;
  static Int_t c__0 = 0;
  static Int_t c__11 = 11;
  static Int_t c__max_nVar_ = max_nVar_;
  static Int_t c__max_nNodes_ = max_nNodes_;
  static Int_t c__12 = 12;
  static Int_t c__999 = 999;
  static Int_t c__13 = 13;

  /* Subroutine */ Int_t train_nn__(Double_t *tin2, Double_t *tout2, Int_t *ntrain, 
				    Int_t *ntest, Int_t *nvar2, Int_t *nlayer, 
				    Int_t *nodes, Int_t *ncycle )
  {
    // sanity checks (AH)
    if (*ntrain + *ntest > max_Events_) {
      printf( "*** TMVA_MethodCFMlpANN_f2c: Warning in train_nn_: number of training + testing" \
	      " events exceeds hardcoded maximum - reset to maximum allowed number");
      *ntrain = *ntrain*(max_Events_/(*ntrain + *ntest));
      *ntest  = *ntest *(max_Events_/(*ntrain + *ntest));
    }
    if (*nvar2 > max_nVar_) {
      printf( "*** TMVA_MethodCFMlpANN_f2c: ERROR in train_nn_: number of variables" \
	      " exceeds hardcoded maximum ==> abort");
      s_stop("", (Int_t)0);
    }
    if (*nlayer > max_nLayers_) {
      printf( "*** TMVA_MethodCFMlpANN_f2c: Warning in train_nn_: number of layers" \
	      " exceeds hardcoded maximum - reset to maximum allowed number");
      *nlayer = max_nLayers_;
    }
    if (*nodes > max_nNodes_) {
      printf( "*** TMVA_MethodCFMlpANN_f2c: Warning in train_nn_: number of nodes" \
	      " exceeds hardcoded maximum - reset to maximum allowed number");
      *nodes = max_nNodes_;
    }

    // create dynamic data tables (AH)
    varn2_1.create( *ntrain + *ntest, *nvar2 );
    varn3_1.create( *ntrain + *ntest, *nvar2 );

    /* System generated locals */
    olist o__1;
    cllist cl__1;

    /* Local variables */
    static Int_t imax;
    extern /* Subroutine */ Int_t test_(), arret_(const char *);
    extern /* Subroutine */ Int_t innit_(char *, Double_t *, Double_t *, 
					 int), entree_new__(Int_t *, char *, Int_t *, Int_t *, 
							    Int_t *, Int_t *, Int_t *, int);
    static char det[20];

    /* ----------------------------------------------------------------------- */
    /*                                                                         */
    /* MultiLayerPerceptron_C : Training code */
    /*                          ------------- */
    /*                                                                         */
    /* authors : J. Proriol and contributions from ALEPH-Clermont-Fd Team members */
    /*                                                                         */
    /* new interface/debug - info collection : P. Gay */
    /* contact : gaypas@afal11.cern.ch */
    /*       Laboratoire                    Universite        IN2P3/ */
    /*           de                         Blaise Pascal     CNRS */
    /*     Physique Corpusculaire */
    /*                                                                         */
    /* ----------------------------------------------------------------------- */
    /*                                                                         */
    /*    NTRAIN: Nb of events used during the learning */
    /*    NTEST: Nb of events used for the test */
    /*    TIN: Input variables */
    /*    TOUT: type of the event */
    /*                                                                         */
    /* ----------------------------------------------------------------------- */

    o__1.oerr = 1;
    o__1.ounit = 30;
    o__1.ofnmlen = 12;
    o__1.ofnm = "mlpl3.weight";
    o__1.orl = 0;
    o__1.osta = "unknown";
    o__1.oacc = 0;
    o__1.ofm = "formatted";
    o__1.oblnk = 0;

    o__1.oerr = 1;
    o__1.ounit = 80;
    o__1.ofnmlen = 9;
    o__1.ofnm = "mlpl3.out";
    o__1.orl = 0;
    o__1.osta = "unknown";
    o__1.oacc = 0;
    o__1.ofm = "formatted";
    o__1.oblnk = 0;

    o__1.oerr = 1;
    o__1.ounit = 48;
    o__1.ofnm = "weights/dummy__.weights";
    o__1.ofnmlen = strlen(o__1.ofnm);
    o__1.orl = 0;
    o__1.osta = "unknown";
    o__1.oacc = 0;
    o__1.ofm = "formatted";
    o__1.oblnk = 0;    

    /*      call entree(det) */
    entree_new__(nvar2, det, ntrain, ntest, nlayer, nodes, ncycle, (Int_t)20);
    if (neur_1.neuron[param_1.layerm - 1] == 1) {
      imax = 2;
      param_1.lclass = 2;
    } else {
      imax = neur_1.neuron[param_1.layerm - 1] << 1;
      param_1.lclass = neur_1.neuron[param_1.layerm - 1];
    }
    param_1.nvar = neur_1.neuron[0];
    test_();
    innit_(det, tout2, tin2, (Int_t)20);
    cl__1.cerr = 0;
    cl__1.cunit = 30;
    cl__1.csta = 0;
    cl__1.cerr = 0;
    cl__1.cunit = 80;
    cl__1.csta = 0;
    cl__1.cerr = 0;
    cl__1.cunit = 48;
    cl__1.csta = 0;
    goto L99;

    arret_(" main routine : mlpl3.weight not opened ");
    goto L99;

    arret_(" main routine : mlpl3.out not opened ");
    goto L99;

    arret_(" main routine : final.weight not opened ");
    goto L99;
  L99:

    // delete data tables
    varn2_1.Delete();
    varn3_1.Delete();

    return 0;
  } /* train_nn__ */

  /* ************************************************************ */
  /* Subroutine */ Int_t entree_new__(Int_t * /*nvar2*/, char * /*det*/, Int_t *ntrain, Int_t *ntest, 
				      Int_t *numlayer, Int_t *nodes, Int_t *numcycle, 
				      Int_t /*det_len*/)
  {
    /* System generated locals */
    Int_t i__1;

    /* Subroutine */ Int_t s_stop(char *, int);

    /* Local variables */
    static Int_t frewrite, i__, j, ncoef;
    extern /* Subroutine */ Int_t arret_(const char *);
    static Int_t ntemp, num, retrain;

    /*    NTRAIN: Nb of events used during the learning */
    /*    NTEST: Nb of events used for the test */
    /*    TIN: Input variables */
    /*    TOUT: type of the event */

    /* if you change the following ranges, you MUST */
    /* do the corresponding changes in "trainvar.inc" */
    /* and "trainvardef.inc" as well !!! */

    /* pg 5-june-98 */
    cost_1.ancout = 1e30;
    /* pg 12-may-98  init */
    /* .............. HardCoded Values .................... */
    retrain = 0;
    frewrite = 1000;
    for (i__ = 1; i__ <= max_nNodes_; ++i__) {
      del_1.coef[i__ - 1] = (Float_t)0.;
    }
    for (i__ = 1; i__ <= max_nLayers_; ++i__) {
      del_1.temp[i__ - 1] = (Float_t)0.;
    }
    param_1.layerm = *numlayer;
    if (param_1.layerm > max_nLayers_) {
      printf("Error: number of layers exceeds maximum: %i, %i ==> abort", 
	     param_1.layerm, max_nLayers_ );
      arret_("modification of mlpl3_param_lim.inc is needed ");
    }
    param_1.nevl = *ntrain;
    param_1.nevt = *ntest;
    param_1.nblearn = *numcycle;
    varn_1.iclass = 2;
    /* pg 8-june-97 - classes naturelles */
    param_1.nunilec = 10;
    /* pg 8-june-97 */
    param_1.epsmin = 1e-10;
    /* pg 8-june-97 */
    param_1.epsmax = 1e-4;
    /* pg 8-june-97 */
    param_1.eta = .5;
    /* pg 8-june-97 */
    cost_1.tolcou = 1e-6;
    /* pg 8-june-97 */
    cost_1.ieps = 2;
    /*     pg nunisor and nunap are fixed ! */
    /* pg 8-june-97 */
    param_1.nunisor = 30;
    /* pg 8-june-97 */
    param_1.nunishort = 48;
    param_1.nunap = 40;
    /* pg 8-june-97 */
    printf("--- TMVA_MethodCFMlpANN: total number of events for training: %i\n", param_1.nevl);
    printf("--- TMVA_MethodCFMlpANN: total number of events for testing : %i\n", param_1.nevt);
    printf("--- TMVA_MethodCFMlpANN: total number of training cycles    : %i\n", param_1.nblearn);
    if (param_1.nevl > max_Events_) {
      printf("Error: number of learning events exceeds maximum: %i, %i ==> abort", 
	     param_1.nevl, max_Events_ );
      arret_("modification of mlpl3_param_lim.inc is needed ");
    }
    if (param_1.nevt > max_Events_) {
      printf("Error: number of testing events exceeds maximum: %i, %i ==> abort", 
	     param_1.nevt, max_Events_ );
      arret_("modification of mlpl3_param_lim.inc is needed ");
    }
    i__1 = param_1.layerm;
    for (j = 1; j <= i__1; ++j) {
      // AH: modified for external configuration;
      // original setting: num = *nvar2 - j + 1;
      num = nodes[j-1];
      if (num < 2) {
	num = 2;
      }
      if (j == param_1.layerm && num != 2) {
	num = 2;
      }
      neur_1.neuron[j - 1] = num;
      /* L50: */
    }
    i__1 = param_1.layerm;
    for (j = 1; j <= i__1; ++j) {
      printf("--- TMVA_MethodCFMlpANN: number of layers for neuron(%2i): %i\n",j, 
	     neur_1.neuron[j - 1]);
      /* nombre de neuron per couche */
    }
    /* Pg */
    if (neur_1.neuron[param_1.layerm - 1] != 2) {
      printf("Error: wrong number of classes at ouput layer: %i != 2 ==> abort\n",
	     neur_1.neuron[param_1.layerm - 1]);
      arret_("stop");
    }
    i__1 = neur_1.neuron[param_1.layerm - 1];
    for (j = 1; j <= i__1; ++j) {
      del_1.coef[j - 1] = 1.;
    }
    i__1 = param_1.layerm;
    for (j = 1; j <= i__1; ++j) {
      del_1.temp[j - 1] = 1.;
    }
    param_1.ichoi = retrain;
    param_1.ndivis = frewrite;
    del_1.idde = 1;
    if (! (param_1.ichoi == 0 || param_1.ichoi == 1)) {
      arret_("new training or continued one !");
    }
    if (param_1.ichoi == 0) {
      printf("--- TMVA_MethodCFMlpANN: new training will be performed\n");
    } 
    else {
      printf("--- TMVA_MethodCFMlpANN: new training will be continued from a weight file\n");
    }
    /*     pg 12 may 98 xcheck */
    ncoef = 0;
    ntemp = 0;
    for (i__ = 1; i__ <= max_nNodes_; ++i__) {
      if (del_1.coef[i__ - 1] != (Float_t)0.) {
	++ncoef;
      }
    }
    for (i__ = 1; i__ <= max_nLayers_; ++i__) {
      if (del_1.temp[i__ - 1] != (Float_t)0.) {
	++ntemp;
      }
    }
    if (ncoef != neur_1.neuron[param_1.layerm - 1]) {
      arret_(" entree error code 1 : need to reported");
    }
    if (ntemp != param_1.layerm) {
      arret_("entree error code 2 : need to reported");
    }
    goto L99;
  L99:
    return 0;
  } /* entree_new__ */

  /* ****************************************************************************** */
  /* Subroutine */ Int_t wini_()
  {
    /* System generated locals */
    Int_t i__1, i__2, i__3;

    /* Local variables */
    extern Double_t sen3a_(Double_t *);
    static Int_t i__, j;
    static Double_t bidon;
    static Int_t layer;

#define w_ref(a_1,a_2,a_3) neur_1.w[((a_3)*max_nNodes_ + (a_2))*max_nLayers_ + a_1 - 187]
#define ww_ref(a_1,a_2) neur_1.ww[(a_2)*max_nLayers_ + a_1 - 7]

    i__1 = param_1.layerm;
    for (layer = 2; layer <= i__1; ++layer) {
      i__2 = neur_1.neuron[layer - 2];
      for (i__ = 1; i__ <= i__2; ++i__) {
	i__3 = neur_1.neuron[layer - 1];
	for (j = 1; j <= i__3; ++j) {
	  w_ref(layer, j, i__) = (sen3a_(&bidon) * 2. - 1.) * .2;
	  ww_ref(layer, j) = (sen3a_(&bidon) * 2. - 1.) * .2;
	  /*     write(6,*)'!!!!!!!! w=',w(layer,j,i) */
	  /* L1: */
	}
      }
    }
    return 0;
  } /* wini_ */

#undef ww_ref
#undef w_ref

  /* *********************************************************************************** */
  /* Subroutine */ Int_t en_ava__(Int_t *ievent)
  {
    /* System generated locals */
    Int_t i__1, i__2, i__3;

    /* Local variables */
    static Double_t f;
    static Int_t i__, j;
    extern /* Subroutine */ Int_t foncf_(Int_t *, Double_t *, Double_t *);
    static Int_t layer;

#define xeev_ref(a_1,a_2) varn2_1(a_1,a_2)
#define w_ref(a_1,a_2,a_3) neur_1.w[((a_3)*max_nNodes_ + (a_2))*max_nLayers_ + a_1 - 187]
#define x_ref(a_1,a_2) neur_1.x[(a_2)*max_nLayers_ + a_1 - 7]
#define y_ref(a_1,a_2) neur_1.y[(a_2)*max_nLayers_ + a_1 - 7]
#define ww_ref(a_1,a_2) neur_1.ww[(a_2)*max_nLayers_ + a_1 - 7]

    i__1 = neur_1.neuron[0];
    for (i__ = 1; i__ <= i__1; ++i__) {
      y_ref(1, i__) = xeev_ref(*ievent, i__);
      /* L1: */
    }
    i__1 = param_1.layerm - 1;
    for (layer = 1; layer <= i__1; ++layer) {
      i__2 = neur_1.neuron[layer];
      for (j = 1; j <= i__2; ++j) {
	x_ref(layer + 1, j) = 0.;
	i__3 = neur_1.neuron[layer - 1];
	for (i__ = 1; i__ <= i__3; ++i__) {
	  x_ref(layer + 1, j) = x_ref(layer + 1, j) + y_ref(layer, i__) 
	    * w_ref(layer + 1, j, i__);
	  /* L3: */
	}
	x_ref(layer + 1, j) = x_ref(layer + 1, j) + ww_ref(layer + 1, j);
	i__3 = layer + 1;
	foncf_(&i__3, &x_ref(layer + 1, j), &f);
	y_ref(layer + 1, j) = f;
	/* L2: */
      }
    }
    return 0;
  } /* en_ava__ */

#undef ww_ref
#undef y_ref
#undef x_ref
#undef w_ref
#undef xeev_ref

  /* ********************************************************************************** */
  /* Subroutine */ Int_t leclearn_(Int_t *ktest, Double_t *tout2, Double_t *
				 tin2)
  {
    /* System generated locals */
    Int_t i__1, i__2;

    /* Local variables */
    static Int_t i__, j, k, l, nocla[max_nNodes_], ikend;
    extern /* Subroutine */ Int_t datacc_(Double_t *, Double_t *, Int_t *,
					  Int_t *, Int_t *, Int_t *, Double_t *, Int_t *, 
					  Int_t *);
    static Double_t xpg[max_nVar_];
    extern /* Subroutine */ Int_t collect_var__(Int_t *, Int_t *, Double_t *);

#define xeev_ref(a_1,a_2) varn2_1(a_1,a_2)

    /*    NTRAIN: Nb of events used during the learning */
    /*    NTEST: Nb of events used for the test */
    /*    TIN: Input variables */
    /*    TOUT: type of the event */

    /* if you change the following ranges, you MUST */
    /* do the corresponding changes in "trainvar.inc" */
    /* and "trainvardef.inc" as well !!! */

    *ktest = 0;
    i__1 = param_1.lclass;
    for (k = 1; k <= i__1; ++k) {
      nocla[k - 1] = 0;
    }
    i__1 = param_1.nvar;
    for (i__ = 1; i__ <= i__1; ++i__) {
      varn_1.xmin[i__ - 1] = 1e30;
      /* L1: */
      varn_1.xmax[i__ - 1] = -varn_1.xmin[i__ - 1];
    }
    i__1 = param_1.nevl;
    for (i__ = 1; i__ <= i__1; ++i__) {
      datacc_(tout2, tin2, &c__100, &c__0, &param_1.nevl, &param_1.nvar, 
	      xpg, &varn_1.nclass[i__ - 1], &ikend);
      if (ikend == -1) {
	goto L3;
      }
      /* pg 20/11/97 */
      collect_var__(&param_1.nvar, &varn_1.nclass[i__ - 1], xpg);
      /* pg 12/06/97 */
      i__2 = param_1.nvar;
      for (j = 1; j <= i__2; ++j) {	
	xeev_ref(i__, j) = xpg[j - 1];
      }
      if (varn_1.iclass == 1) {
	i__2 = param_1.lclass;
	for (k = 1; k <= i__2; ++k) {
	  if (varn_1.nclass[i__ - 1] == k) {
	    ++nocla[k - 1];
	  }
	  /* L20: */
	}
      }
      i__2 = param_1.nvar;
      for (k = 1; k <= i__2; ++k) {
	if (xeev_ref(i__, k) < varn_1.xmin[k - 1]) {
	  varn_1.xmin[k - 1] = xeev_ref(i__, k);
	}
	if (xeev_ref(i__, k) > varn_1.xmax[k - 1]) {
	  varn_1.xmax[k - 1] = xeev_ref(i__, k);
	}
	/* L2: */
      }
    }
  L3:
    /* pg 20/11/97 */
    if (varn_1.iclass == 1) {
      i__2 = param_1.lclass;
      for (k = 1; k <= i__2; ++k) {
	i__1 = param_1.lclass;
	for (l = 1; l <= i__1; ++l) {
	  if (nocla[k - 1] != nocla[l - 1]) {
	    *ktest = 1;
	  }
	  /* L40: */
	}
      }
    }
    i__1 = param_1.nevl;
    for (i__ = 1; i__ <= i__1; ++i__) {
      i__2 = param_1.nvar;
      for (l = 1; l <= i__2; ++l) {
	if (varn_1.xmax[l - 1] == (Float_t)0. && varn_1.xmin[l - 1] == (
								      Float_t)0.) {
	  xeev_ref(i__, l) = (Float_t)0.;
	} else {
	  xeev_ref(i__, l) = xeev_ref(i__, l) - (varn_1.xmax[l - 1] + 
						 varn_1.xmin[l - 1]) / 2.;
	  xeev_ref(i__, l) = xeev_ref(i__, l) / ((varn_1.xmax[l - 1] - 
						  varn_1.xmin[l - 1]) / 2.);
	}
	/* L30: */
      }
    }
    return 0;
  } /* leclearn_ */

#undef xeev_ref

  /* ************************************************************************************ */
  /* Subroutine */ Int_t en_arr__(Int_t *ievent)
  {
    /* System generated locals */
    Int_t i__1, i__2, i__3;

    /* Local variables */
    static Double_t f;
    static Int_t i__, j, k, l;
    extern /* Subroutine */ Int_t foncf_(Int_t *, Double_t *, Double_t *);
    static Double_t df, uu;

#define delw_ref(a_1,a_2,a_3) del_1.delw[((a_3)*max_nNodes_ + (a_2))*max_nLayers_ + a_1 - 187]
#define w_ref(a_1,a_2,a_3) neur_1.w[((a_3)*max_nNodes_ + (a_2))*max_nLayers_ + a_1 - 187]
#define x_ref(a_1,a_2) neur_1.x[(a_2)*max_nLayers_ + a_1 - 7]
#define y_ref(a_1,a_2) neur_1.y[(a_2)*max_nLayers_ + a_1 - 7]
#define delta_ref(a_1,a_2,a_3) del_1.delta[((a_3)*max_nNodes_ + (a_2))*max_nLayers_ + a_1 - \
					   187]
#define delww_ref(a_1,a_2) del_1.delww[(a_2)*max_nLayers_ + a_1 - 7]
#define ww_ref(a_1,a_2) neur_1.ww[(a_2)*max_nLayers_ + a_1 - 7]
#define del_ref(a_1,a_2) del_1.del[(a_2)*max_nLayers_ + a_1 - 7]
#define deltaww_ref(a_1,a_2) neur_1.deltaww[(a_2)*max_nLayers_ + a_1 - 7]

    i__1 = neur_1.neuron[param_1.layerm - 1];
    for (i__ = 1; i__ <= i__1; ++i__) {
      if (varn_1.nclass[*ievent - 1] == i__) {
	neur_1.o[i__ - 1] = 1.;
      } else {
	neur_1.o[i__ - 1] = -1.;
      }
      /* L10: */
    }
    l = param_1.layerm;
    i__1 = neur_1.neuron[l - 1];
    for (i__ = 1; i__ <= i__1; ++i__) {
      f = y_ref(l, i__);
      df = (f + 1.) * (1. - f) / (del_1.temp[l - 1] * 2.);
      del_ref(l, i__) = df * (neur_1.o[i__ - 1] - y_ref(l, i__)) * 
	del_1.coef[i__ - 1];
      delww_ref(l, i__) = param_1.eeps * del_ref(l, i__);
      i__2 = neur_1.neuron[l - 2];
      for (j = 1; j <= i__2; ++j) {
	delw_ref(l, i__, j) = param_1.eeps * del_ref(l, i__) * y_ref(l - 
								     1, j);
	/* L20: */
      }
    }
    for (l = param_1.layerm - 1; l >= 2; --l) {
      i__2 = neur_1.neuron[l - 1];
      for (i__ = 1; i__ <= i__2; ++i__) {
	uu = 0.;
	i__1 = neur_1.neuron[l];
	for (k = 1; k <= i__1; ++k) {
	  uu += w_ref(l + 1, k, i__) * del_ref(l + 1, k);
	  /* L40: */
	}
	foncf_(&l, &x_ref(l, i__), &f);
	df = (f + 1.) * (1. - f) / (del_1.temp[l - 1] * 2.);
	del_ref(l, i__) = df * uu;
	delww_ref(l, i__) = param_1.eeps * del_ref(l, i__);
	i__1 = neur_1.neuron[l - 2];
	for (j = 1; j <= i__1; ++j) {
	  delw_ref(l, i__, j) = param_1.eeps * del_ref(l, i__) * y_ref(
								       l - 1, j);
	  /* L30: */
	}
      }
    }
    i__1 = param_1.layerm;
    for (l = 2; l <= i__1; ++l) {
      i__2 = neur_1.neuron[l - 1];
      for (i__ = 1; i__ <= i__2; ++i__) {
	deltaww_ref(l, i__) = delww_ref(l, i__) + param_1.eta * 
	  deltaww_ref(l, i__);
	ww_ref(l, i__) = ww_ref(l, i__) + deltaww_ref(l, i__);
	i__3 = neur_1.neuron[l - 2];
	for (j = 1; j <= i__3; ++j) {
	  delta_ref(l, i__, j) = delw_ref(l, i__, j) + param_1.eta * 
	    delta_ref(l, i__, j);
	  w_ref(l, i__, j) = w_ref(l, i__, j) + delta_ref(l, i__, j);
	  /* L50: */
	}
      }
    }
    return 0;
  } /* en_arr__ */

#undef deltaww_ref
#undef del_ref
#undef ww_ref
#undef delww_ref
#undef delta_ref
#undef y_ref
#undef x_ref
#undef w_ref
#undef delw_ref

  /* ******************************************************************************* */
  /* Subroutine */ Int_t out_(Int_t *iii, Int_t *maxcycle)
  {

    /* Local variables */

    // external function
    extern void writeWeightFile_( Int_t, Int_t, 
				  Double_t*, Double_t*, 
				  Int_t, int*, 
				  Double_t*, Double_t*, Double_t* );

#define w_ref(a_1,a_2,a_3) neur_1.w[((a_3)*max_nNodes_ + (a_2))*max_nLayers_ + a_1 - 187]
#define ww_ref(a_1,a_2) neur_1.ww[(a_2)*max_nLayers_ + a_1 - 7]

    /*    NTRAIN: Nb of events used during the learning */
    /*    NTEST : Nb of events used for the test */
    /*    TIN   : Input variables */
    /*    TOUT  : type of the event */

    // AH: call write routine in main class (to use C++ I/O functionality)
    if (*iii == *maxcycle) {
      writeWeightFile_( param_1.nvar, param_1.lclass, 
			varn_1.xmax, varn_1.xmin,
			param_1.layerm, neur_1.neuron, 
			neur_1.w, neur_1.ww, del_1.temp );			
    }

    return 0;
  } /* out_ */

#undef ww_ref
#undef w_ref

  /* **************************************************************************************** */
  /* Subroutine */ Int_t innit_(char *det, Double_t *tout2, Double_t *tin2, 
				Int_t  /*det_len*/)
  {
    /* Format strings */

    /* System generated locals */
    Int_t i__1, i__2, i__3;

    /* Builtin functions */
    /* Subroutine */ Int_t s_stop(char *, int);

    /* Local variables */
    extern /* Subroutine */ Int_t leclearn_(Int_t *, Double_t *, Double_t 
					  *), wini_();
    static Float_t btup[3];
    extern /* Subroutine */ Int_t cout_(Int_t *, Double_t *);
    extern Double_t sen3a_(Double_t *);
    extern /* Subroutine */ Int_t cout2_(Int_t *, Double_t *);
    static Int_t i__, j;
    static Double_t bidon;
    extern /* Subroutine */ Int_t graph_(Int_t *, Double_t *, Double_t *, 
				       char *, int);
    static Int_t nevod, layer, ktest, i1, nrest;
    extern /* Subroutine */ Int_t lecev2_(Int_t *, Double_t *, Double_t *)
      , en_ava__(Int_t *), en_arr__(Int_t *);
    static Int_t ievent;
    extern /* Subroutine */ Int_t histow_(), hfn_(Int_t *, Float_t *);
    static Int_t kkk;
    extern /* Subroutine */ Int_t inl_();
    extern Double_t fdecroi_(Int_t *);
    extern /* Subroutine */ Int_t out_(Int_t *, Int_t *);
    static Double_t xxx, yyy;

#define delta_ref(a_1,a_2,a_3) del_1.delta[((a_3)*max_nNodes_ + (a_2))*max_nLayers_ + a_1 - 187]
#define deltaww_ref(a_1,a_2) neur_1.deltaww[(a_2)*max_nLayers_ + a_1 - 7]

    /*    NTRAIN: Nb of events used during the learning */
    /*    NTEST: Nb of events used for the test */
    /*    TIN: Input variables */
    /*    TOUT: type of the event */

    /* if you change the following ranges, you MUST */
    /* do the corresponding changes in "trainvar.inc" */
    /* and "trainvardef.inc" as well !!! */

    leclearn_(&ktest, tout2, tin2);
    lecev2_(&ktest, tout2, tin2);
    if (ktest == 1) {
      s_stop("", (Int_t)0);
    }
    i__1 = param_1.layerm - 1;
    for (layer = 1; layer <= i__1; ++layer) {
      i__2 = neur_1.neuron[layer];
      for (j = 1; j <= i__2; ++j) {
	deltaww_ref(layer + 1, j) = 0.;
	i__3 = neur_1.neuron[layer - 1];
	for (i__ = 1; i__ <= i__3; ++i__) {
	  delta_ref(layer + 1, j, i__) = 0.;
	  /* L300: */
	}
      }
    }
    if (param_1.ichoi == 1) {
      inl_();
    } else {
      wini_();
    }
    kkk = 0;
    i__3 = param_1.nblearn;
    TMVA_Timer timer( i__3, "TMVA_MethodCFMlpANN" ); 
    Int_t num = i__3/100;

    for (i1 = 1; i1 <= i__3; ++i1) {

      if ((i1-1)%num == 0 || i1 == i__3) timer.DrawProgressBar( i1-1 );

      i__2 = param_1.nevl;
      for (i__ = 1; i__ <= i__2; ++i__) {
	++kkk;
	if (cost_1.ieps == 2) {
	  param_1.eeps = fdecroi_(&kkk);
	}
	if (cost_1.ieps == 1) {
	  param_1.eeps = param_1.epsmin;
	}
	if (varn_1.iclass == 2) {
	  ievent = (Int_t) ((Double_t) param_1.nevl * sen3a_(&bidon)
			      );
	  if (ievent == 0) {
	    goto L1;
	  }
	}
	if (varn_1.iclass == 1) {
	  nevod = param_1.nevl / param_1.lclass;
	  nrest = i__ % param_1.lclass;
	  param_1.ndiv = i__ / param_1.lclass;
	  if (nrest != 0) {
	    ievent = param_1.ndiv + 1 + (param_1.lclass - nrest) * 
	      nevod;
	  } else {
	    ievent = param_1.ndiv;
	  }
	}
	en_ava__(&ievent);
	en_arr__(&ievent);
      L1:
	;
      }
      yyy = 0.;
      if (i1 % param_1.ndivis == 0 || i1 == 1 || i1 == param_1.nblearn) {
	cout_(&i1, &xxx);
	cout2_(&i1, &yyy);
	graph_(&i1, &xxx, &yyy, det, (Int_t)20);
	out_(&i1, &param_1.nblearn);
	btup[0] = (Float_t) i1;
	btup[1] = (Float_t) xxx;
	btup[2] = (Float_t) yyy;
	if (nnfillntuplecom_1.nnfillntuple) {
	  hfn_(&c__11, btup);
	}
      }
      if (xxx < cost_1.tolcou) {
	graph_(&param_1.nblearn, &xxx, &yyy, det, (Int_t)20);
	out_(&param_1.nblearn, &param_1.nblearn);
	goto L70;
      }
      /* L60: */
    }
    printf("--- TMVA_MethodCFMlpANN: elapsed time: %s\n", (const char*)timer.GetElapsedTime()  );
  L70:
    histow_();
    return 0;
  } /* innit_ */

#undef deltaww_ref
#undef delta_ref

  /* ************************************************************************************* */
  /* Subroutine */ Int_t test_()
  {
    /* System generated locals */
    Int_t i__1;

    /* Subroutine */ Int_t s_stop(char *, int);

    /* Local variables */
    static Int_t i__;
    extern /* Subroutine */ Int_t arret_(const char *);
    static Int_t ktest;

    /* Fortran I/O blocks */

    ktest = 0;
    if (param_1.layerm > max_nLayers_) {
      ktest = 1;
      printf("Error: number of layers exceeds maximum: %i, %i ==> abort", 
	     param_1.layerm, max_nLayers_ );
      arret_("modification of mlpl3_param_lim.inc is needed ");
    }
    if (param_1.nevl > max_Events_) {
      ktest = 1;
      printf("Error: number of training events exceeds maximum: %i, %i ==> abort", 
	     param_1.nevl, max_Events_ );
      arret_("modification of mlpl3_param_lim.inc is needed ");
    }
    if (param_1.nevt > max_Events_) {
      printf("Error: number of testing events exceeds maximum: %i, %i ==> abort", 
	     param_1.nevt, max_Events_ );
      arret_("modification of mlpl3_param_lim.inc is needed ");
    }
    if (param_1.lclass < neur_1.neuron[param_1.layerm - 1]) {
      ktest = 1;
      printf("Error: wrong number of classes at ouput layer: %i != %i ==> abort\n",
	     neur_1.neuron[param_1.layerm - 1], param_1.lclass);
      arret_("problem needs to reported ");
    }
    if (param_1.nvar > max_nVar_) {
      ktest = 1;
      printf("Error: number of variables exceeds maximum: %i, %i ==> abort", 
	     param_1.nvar, c__max_nVar_ );
      arret_("modification of mlpl3_param_lim.inc is needed");
    }
    i__1 = param_1.layerm;
    for (i__ = 1; i__ <= i__1; ++i__) {
      if (neur_1.neuron[i__ - 1] > max_nNodes_) {
	ktest = 1;
	printf("Error: number of neurons at layer exceeds maximum: %i, %i ==> abort", 
	       i__, c__max_nNodes_ );
      }
    }
    if (ktest == 1) {
      s_stop("", (Int_t)0);
    }
    return 0;
  } /* test_ */

  /* ****************************************************************************************** */
  /* Subroutine */ Int_t cout_(Int_t * /*i1*/, Double_t *xxx)
  {
    /* System generated locals */
    Int_t i__1, i__2;
    Double_t d__1;

    /* Local variables */
    static Double_t c__;
    static Int_t i__, j;
    extern /* Subroutine */ Int_t en_ava__(Int_t *);

    /* Fortran I/O blocks */

#define y_ref(a_1,a_2) neur_1.y[(a_2)*max_nLayers_ + a_1 - 7]

    c__ = 0.;
    i__1 = param_1.nevl;
    for (i__ = 1; i__ <= i__1; ++i__) {
      en_ava__(&i__);
      i__2 = neur_1.neuron[param_1.layerm - 1];
      for (j = 1; j <= i__2; ++j) {
	if (varn_1.nclass[i__ - 1] == j) {
	  neur_1.o[j - 1] = 1.;
	} else {
	  neur_1.o[j - 1] = -1.;
	}
	/* Computing 2nd power */
	d__1 = y_ref(param_1.layerm, j) - neur_1.o[j - 1];
	c__ += del_1.coef[j - 1] * (d__1 * d__1);
	/* L10: */
      }
    }
    c__ /= (Double_t) (param_1.nevl * param_1.lclass) * 2.;
    /*     write(80,*)'cost(',i1,')=',c */
    *xxx = c__;
    cost_1.ancout = c__;
    return 0;
  } /* cout_ */

#undef y_ref


  /* ************************************************************************************* */
  /* Subroutine */ Int_t inl_()
  {
    /* Initialized data */

    static Int_t init = 0;

    /* Format strings */
    static char fmt_10[] = "(2i8)";
    static char fmt_12[] = "(2e15.8)";
    static char fmt_3[] = "(10i8)";
    static char fmt_2[] = "(10e12.5)";
    static char fmt_16[] = "(/)";
    static char fmt_4[] = "(/,/,e12.5)";

    /* System generated locals */
    Int_t i__1, i__2, i__3;
    olist o__1;

    /* Subroutine */ Int_t s_stop(char *, int);

    /* Local variables */
    static Int_t jmin, jmax, i__, k, layer, kk, nq, nr;

    /* Fortran I/O blocks */
    static cilist io___127 = { 0, 0, 0, fmt_10, 0 };
    static cilist io___129 = { 0, 0, 0, fmt_12, 0 };
    static cilist io___130 = { 0, 0, 0, fmt_10, 0 };
    static cilist io___131 = { 0, 0, 0, fmt_3, 0 };
    static cilist io___140 = { 0, 0, 0, fmt_2, 0 };
    static cilist io___141 = { 0, 0, 0, fmt_2, 0 };
    static cilist io___142 = { 0, 0, 0, fmt_16, 0 };
    static cilist io___143 = { 0, 0, 0, fmt_4, 0 };

#define w_ref(a_1,a_2,a_3) neur_1.w[((a_3)*max_nNodes_ + (a_2))*max_nLayers_ + a_1 - 187]
#define ww_ref(a_1,a_2) neur_1.ww[(a_2)*max_nLayers_ + a_1 - 7]

    if (init == 0) {
      o__1.oerr = 1;
      o__1.ounit = param_1.nunap;
      o__1.ofnmlen = 9;
      o__1.ofnm = "mlpl3.res";
      o__1.orl = 0;
      o__1.osta = "old";
      o__1.oacc = 0;
      o__1.ofm = "formatted";
      o__1.oblnk = 0;
    }
    io___127.ciunit = param_1.nunap;
    i__1 = param_1.nvar;
    for (i__ = 1; i__ <= i__1; ++i__) {
      io___129.ciunit = param_1.nunap;
      /* L11: */
    }
    io___130.ciunit = param_1.nunap;
    io___131.ciunit = param_1.nunap;
    i__1 = param_1.layerm;
    i__1 = param_1.layerm - 1;
    for (layer = 1; layer <= i__1; ++layer) {
      nq = neur_1.neuron[layer] / 10;
      nr = neur_1.neuron[layer] - nq * 10;
      if (nr == 0) {
	kk = nq;
      } else {
	kk = nq + 1;
      }
      i__2 = kk;
      for (k = 1; k <= i__2; ++k) {
	jmin = k * 10 - 9;
	jmax = k * 10;
	if (neur_1.neuron[layer] < jmax) {
	  jmax = neur_1.neuron[layer];
	}
	io___140.ciunit = param_1.nunap;
	i__3 = neur_1.neuron[layer - 1];
	for (i__ = 1; i__ <= i__3; ++i__) {
	  io___141.ciunit = param_1.nunap;
	  /* L1: */
	}
      }
      io___142.ciunit = param_1.nunap;
      /* L15: */
    }
    io___143.ciunit = param_1.nunap;
    goto L99;

    printf(" error while opening mlpl3.res" );
    s_stop("", (Int_t)0);
  L99:
    return 0;
  } /* inl_ */

#undef ww_ref
#undef w_ref

  /* **************************************************************************************** */
  Double_t fdecroi_(Int_t *i__)
  {
    /* System generated locals */
    Double_t ret_val;

    /* Local variables */
    static Double_t aaa, bbb;

    aaa = (param_1.epsmin - param_1.epsmax) / (Double_t) (param_1.nblearn * 
							    param_1.nevl - 1);
    bbb = param_1.epsmax - aaa;
    ret_val = aaa * (Double_t) (*i__) + bbb;
    return ret_val;
  } /* fdecroi_ */

  /* ***************************************************************************************** */
  /* Subroutine */ Int_t graph_(Int_t *ilearn, Double_t * /*xxx*/, Double_t * /*yyy*/,
				char * /*det*/, Int_t  /*det_len*/)
  {
    /* Format strings */

    /* System generated locals */
    Int_t i__1, i__2;

    /* Local variables */
    static Double_t xmok[max_nNodes_];
    static Float_t xpaw;
    static Double_t xmko[max_nNodes_];
    static Int_t i__, j;
    extern /* Subroutine */ Int_t en_ava__(Int_t *);
    static Int_t ix;
    extern /* Subroutine */ Int_t hfn_(Int_t *, Float_t *);
    static Int_t jjj;
    static Float_t vbn[10];
    static Int_t nko[max_nNodes_], nok[max_nNodes_];

    /* Fortran I/O blocks */

#define y_ref(a_1,a_2) neur_1.y[(a_2)*max_nLayers_ + a_1 - 7]

    for (i__ = 1; i__ <= 10; ++i__) {
      vbn[i__ - 1] = (Float_t)0.;
    }
    if (*ilearn == 1) {
      // AH: removed output 
    }
    i__1 = neur_1.neuron[param_1.layerm - 1];
    for (i__ = 1; i__ <= i__1; ++i__) {
      nok[i__ - 1] = 0;
      nko[i__ - 1] = 0;
      xmok[i__ - 1] = 0.;
      xmko[i__ - 1] = 0.;
      /* L100: */
    }
    i__1 = param_1.nevl;
    for (i__ = 1; i__ <= i__1; ++i__) {
      en_ava__(&i__);
      i__2 = neur_1.neuron[param_1.layerm - 1];
      for (j = 1; j <= i__2; ++j) {
	xpaw = (Float_t) y_ref(param_1.layerm, j);
	if (varn_1.nclass[i__ - 1] == j) {
	  ++nok[j - 1];
	  xmok[j - 1] += y_ref(param_1.layerm, j);
	} else {
	  ++nko[j - 1];
	  xmko[j - 1] += y_ref(param_1.layerm, j);
	  jjj = j + neur_1.neuron[param_1.layerm - 1];
	}
	if (j <= 9) {
	  vbn[j - 1] = xpaw;
	}
	/* pg */
	/* L150: */
      }
      vbn[9] = (Float_t) varn_1.nclass[i__ - 1];
      /* pg */
      if (*ilearn == param_1.nblearn) {
	if (nnfillntuplecom_1.nnfillntuple) {
	  hfn_(&c__12, vbn);
	}
	/* pg */
      }
      /* L10: */
    }
    i__1 = neur_1.neuron[param_1.layerm - 1];
    for (j = 1; j <= i__1; ++j) {
      xmok[j - 1] /= (Double_t) nok[j - 1];
      xmko[j - 1] /= (Double_t) nko[j - 1];
      neur_1.cut[j - 1] = (xmok[j - 1] + xmko[j - 1]) / 2.;
      /* L200: */
    }
    if (*ilearn == 1) {
      // AH: removed output
    }
    ix = neur_1.neuron[param_1.layerm - 1];
    i__1 = ix;

    return 0;
  } /* graph_ */

#undef y_ref

  /* ****************************************************************************************** */
  Double_t sen3a_(Double_t * /*bidon*/)
  {
    /* Initialized data */

    static Int_t m12 = 4096;
    static Double_t f1 = 2.44140625e-4;
    static Double_t f2 = 5.96046448e-8;
    static Double_t f3 = 1.45519152e-11;
    static Int_t j1 = 3823;
    static Int_t j2 = 4006;
    static Int_t j3 = 2903;
    static Int_t i1 = 3823;
    static Int_t i2 = 4006;
    static Int_t i3 = 2903;

    /* System generated locals */
    Double_t ret_val;

    /* Local variables */
    static Int_t k3, l3, k2, l2, k1, l1;

    /*     reference /k.d.senne/j. stochastics/ vol 1,no 3 (1974),pp.215-38/  * */
    k3 = i3 * j3;
    l3 = k3 / m12;
    k2 = i2 * j3 + i3 * j2 + l3;
    l2 = k2 / m12;
    k1 = i1 * j3 + i2 * j2 + i3 * j1 + l2;
    l1 = k1 / m12;
    i1 = k1 - l1 * m12;
    i2 = k2 - l2 * m12;
    i3 = k3 - l3 * m12;
    ret_val = f1 * (Double_t) i1 + f2 * (Float_t) i2 + f3 * (Double_t) i3;
    return ret_val;
  } /* sen3a_ */

  /* *************************************************************************************** */
  /* Subroutine */ Int_t foncf_(Int_t *i__, Double_t *u, Double_t *f)
  {
    /* Builtin functions */
    Double_t exp(Double_t);

    /* Local variables */
    static Double_t yy;

    if (*u / del_1.temp[*i__ - 1] > 170.) {
      *f = .99999999989999999;
    } else if (*u / del_1.temp[*i__ - 1] < -170.) {
      *f = -.99999999989999999;
    } else {
      yy = exp(-(*u) / del_1.temp[*i__ - 1]);
      *f = (1. - yy) / (yy + 1.);
    }
    return 0;
  } /* foncf_ */

  /* **************************************************************************************** */
  /* Subroutine */ Int_t histow_()
  {
    /* System generated locals */
    Int_t i__1, i__2, i__3;

    /* Local variables */
    static Double_t wmin, wmax;
    static Int_t i__, j, l;
    static Double_t www;

#define w_ref(a_1,a_2,a_3) neur_1.w[((a_3)*max_nNodes_ + (a_2))*max_nLayers_ + a_1 - 187]

    wmin = 1e30;
    wmax = -wmin;
    i__1 = param_1.layerm - 1;
    for (l = 1; l <= i__1; ++l) {
      i__2 = neur_1.neuron[l];
      for (i__ = 1; i__ <= i__2; ++i__) {
	i__3 = neur_1.neuron[l - 1];
	for (j = 1; j <= i__3; ++j) {
	  if (w_ref(l + 1, i__, j) < wmin) {
	    wmin = w_ref(l + 1, i__, j);
	  }
	  if (w_ref(l + 1, i__, j) > wmax) {
	    wmax = w_ref(l + 1, i__, j);
	  }
	  /* L10: */
	}
      }
    }
    i__3 = param_1.layerm - 1;
    for (l = 1; l <= i__3; ++l) {
      i__2 = neur_1.neuron[l];
      for (i__ = 1; i__ <= i__2; ++i__) {
	i__1 = neur_1.neuron[l - 1];
	for (j = 1; j <= i__1; ++j) {
	  www = w_ref(l + 1, i__, j) - (wmax + wmin) / 2.;
	  www /= (wmax - wmin) / 2.;
	  /* PG                      call hfill(100,www,0.,1.) */
	  /* L20: */
	}
      }
    }
    return 0;
  } /* histow_ */

#undef w_ref


  /* ***************************************************************************************** */
  /* Subroutine */ Int_t cout2_(Int_t * /*i1*/, Double_t *yyy)
  {
    /* System generated locals */
    Int_t i__1, i__2;
    Double_t d__1;

    /* Local variables */
    static Double_t c__;
    static Int_t i__, j;
    extern /* Subroutine */ Int_t en_av2__(Int_t *);

#define y_ref(a_1,a_2) neur_1.y[(a_2)*max_nLayers_ + a_1 - 7]

    c__ = 0.;
    i__1 = param_1.nevt;
    for (i__ = 1; i__ <= i__1; ++i__) {
      en_av2__(&i__);
      i__2 = neur_1.neuron[param_1.layerm - 1];
      for (j = 1; j <= i__2; ++j) {
	if (varn_1.mclass[i__ - 1] == j) {
	  neur_1.o[j - 1] = 1.;
	} else {
	  neur_1.o[j - 1] = -1.;
	}
	/* Computing 2nd power */
	d__1 = y_ref(param_1.layerm, j) - neur_1.o[j - 1];
	c__ += del_1.coef[j - 1] * (d__1 * d__1);
	/* L10: */
      }
    }
    c__ /= (Double_t) (param_1.nevt * param_1.lclass) * 2.;
    /*     write(80,*)'cost(',i1,')=',c */
    *yyy = c__;
    return 0;
  } /* cout2_ */

#undef y_ref

  /* ****************************************************************************************** */
  /* Subroutine */ Int_t lecev2_(Int_t *ktest, Double_t *tout2, Double_t *
			       tin2)
  {
    /* System generated locals */
    Int_t i__1, i__2;
    cllist cl__1;

    /* Local variables */
    static Int_t i__, j, k, l, mocla[max_nNodes_], ikend;
    extern /* Subroutine */ Int_t datacc_(Double_t *, Double_t *, Int_t *,
					  Int_t *, Int_t *, Int_t *, Double_t *, Int_t *, 
					  Int_t *);
    static Double_t xpg[max_nVar_];

#define xx_ref(a_1,a_2) varn3_1(a_1,a_2)

    /*    NTRAIN: Nb of events used during the learning */
    /*    NTEST: Nb of events used for the test */
    /*    TIN: Input variables */
    /*    TOUT: type of the event */

    /* if you change the following ranges, you MUST */
    /* do the corresponding changes in "trainvar.inc" */
    /* and "trainvardef.inc" as well !!! */

    *ktest = 0;
    i__1 = param_1.lclass;
    for (k = 1; k <= i__1; ++k) {
      mocla[k - 1] = 0;
    }
    i__1 = param_1.nevt;
    for (i__ = 1; i__ <= i__1; ++i__) {
      datacc_(tout2, tin2, &c__999, &c__0, &param_1.nevt, &param_1.nvar, 
	      xpg, &varn_1.mclass[i__ - 1], &ikend);
      /* p */
      if (ikend == -1) {
	goto L3;
      }
      /* pg 20/11/97 */
      i__2 = param_1.nvar;
      for (j = 1; j <= i__2; ++j) {
	xx_ref(i__, j) = xpg[j - 1];
      }
    }
  L3:
    /* pg 20/11/97 */
    i__1 = param_1.nevt;
    for (i__ = 1; i__ <= i__1; ++i__) {
      i__2 = param_1.nvar;
      for (l = 1; l <= i__2; ++l) {
	if (varn_1.xmax[l - 1] == (Float_t)0. && varn_1.xmin[l - 1] == (
								      Float_t)0.) {
	  xx_ref(i__, l) = (Float_t)0.;
	} else {
	  xx_ref(i__, l) = xx_ref(i__, l) - (varn_1.xmax[l - 1] + 
					     varn_1.xmin[l - 1]) / 2.;
	  xx_ref(i__, l) = xx_ref(i__, l) / ((varn_1.xmax[l - 1] - 
					      varn_1.xmin[l - 1]) / 2.);
	}
      }
    }
    cl__1.cerr = 0;
    cl__1.cunit = 11;
    cl__1.csta = 0;

    return 0;
  } /* lecev2_ */

#undef xx_ref


  /* ***************************************************************************************** */
  /* Subroutine */ Int_t en_av2__(Int_t *ievent)
  {
    /* System generated locals */
    Int_t i__1, i__2, i__3;

    /* Local variables */
    static Double_t f;
    static Int_t i__, j;
    extern /* Subroutine */ Int_t foncf_(Int_t *, Double_t *, Double_t *);
    static Int_t layer;

#define w_ref(a_1,a_2,a_3) neur_1.w[((a_3)*max_nNodes_ + (a_2))*max_nLayers_ + a_1 - 187]
#define x_ref(a_1,a_2) neur_1.x[(a_2)*max_nLayers_ + a_1 - 7]
#define y_ref(a_1,a_2) neur_1.y[(a_2)*max_nLayers_ + a_1 - 7]
#define ww_ref(a_1,a_2) neur_1.ww[(a_2)*max_nLayers_ + a_1 - 7]
#define xx_ref(a_1,a_2) varn3_1(a_1,a_2)

    i__1 = neur_1.neuron[0];
    for (i__ = 1; i__ <= i__1; ++i__) {
      y_ref(1, i__) = xx_ref(*ievent, i__);
      /* L1: */
    }
    i__1 = param_1.layerm - 1;
    for (layer = 1; layer <= i__1; ++layer) {
      i__2 = neur_1.neuron[layer];
      for (j = 1; j <= i__2; ++j) {
	x_ref(layer + 1, j) = 0.;
	i__3 = neur_1.neuron[layer - 1];
	for (i__ = 1; i__ <= i__3; ++i__) {
	  x_ref(layer + 1, j) = x_ref(layer + 1, j) + y_ref(layer, i__) 
	    * w_ref(layer + 1, j, i__);
	  /* L3: */
	}
	x_ref(layer + 1, j) = x_ref(layer + 1, j) + ww_ref(layer + 1, j);
	i__3 = layer + 1;
	foncf_(&i__3, &x_ref(layer + 1, j), &f);
	y_ref(layer + 1, j) = f;
	/* L2: */
      }
    }
    return 0;
  } /* en_av2__ */

#undef xx_ref
#undef ww_ref
#undef y_ref
#undef x_ref
#undef w_ref


  /* *************************************************************************************** */
  /* Subroutine */ Int_t arret_(const char* mot )
  {
    /* Builtin functions */
    printf("TMVA_MethodCFMlpANN: %s",mot);
    s_stop("STOP", (Int_t)4);
    return 0;
  } /* arret_ */

  /* ***************************************************************************************** */
  /* Subroutine */ Int_t collect_var__(Int_t *nvar, Int_t *class__, 
				     Double_t *xpg)
  {
    /* System generated locals */
    Int_t i__1;

    /* Local variables */
    static Int_t i__;
    static Float_t x[201];
    extern /* Subroutine */ Int_t hfn_(Int_t *, Float_t *);

    /* Parameter adjustments */
    --xpg;

    /* Function Body */
    for (i__ = 1; i__ <= 201; ++i__) {
      x[i__ - 1] = (Float_t)0.;
    }
    x[0] = (Float_t) (*class__);
    i__1 = *nvar;
    for (i__ = 1; i__ <= i__1; ++i__) {
      x[i__] = (Float_t) xpg[i__];
    }
    if (nnfillntuplecom_1.nnfillntuple) {
      hfn_(&c__13, x);
    }
    return 0;
  } /* collect_var__ */

  /* ****************************************************************************************** */
  /* Subroutine */ 

  Int_t hfn_(Int_t * /*id*/, Float_t * /*xtup */)
  {
    // dummy routine at present; fill with Root interface
    return 0;
  }

#ifdef __cplusplus
}
#endif
#endif
