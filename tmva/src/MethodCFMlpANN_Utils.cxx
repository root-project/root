// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::MethodCFMlpANN_utils                                            *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Reference for the original FORTRAN version "mlpl3.F":                          *
 *      Authors  : J. Proriol and contributions from ALEPH-Clermont-Ferrand       *
 *                 Team members                                                   *
 *      Copyright: Laboratoire Physique Corpusculaire                             *
 *                 Universite de Blaise Pascal, IN2P3/CNRS                        *
 *                                                                                *
 * Modifications by present authors:                                              *
 *      use dynamical data tables (not for all of them, but for the big ones)     *
 *                                                                                *
 * Description:                                                                   *
 *      Utility routine translated from original mlpl3.F FORTRAN routine          *
 *                                                                                *
 *      MultiLayerPerceptron : Training code                                      *
 *                                                                                *
 *        NTRAIN: Nb of events used during the learning                           *
 *        NTEST:  Nb of events used for the test                                  *
 *        TIN:    Input variables                                                 *
 *        TOUT:   type of the event                                               *
 *                                                                                *
 *  ----------------------------------------------------------------------------  *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 *                                                                                *
 **********************************************************************************/

//_______________________________________________________________________
//
// Implementation of Clermond-Ferrand artificial neural network
//
// Reference for the original FORTRAN version "mlpl3.F":
//      Authors  : J. Proriol and contributions from ALEPH-Clermont-Ferrand
//                 Team members
//      Copyright: Laboratoire Physique Corpusculaire
//                 Universite de Blaise Pascal, IN2P3/CNRS
//_______________________________________________________________________

#include <string>
#include <iostream>
#include <cstdlib>

#include "TMath.h"
#include "TString.h"

#include "TMVA/MethodCFMlpANN_Utils.h"
#include "TMVA/Timer.h"

using std::cout;
using std::endl;

ClassImp(TMVA::MethodCFMlpANN_Utils)
   
Int_t       TMVA::MethodCFMlpANN_Utils::fg_100         = 100;
Int_t       TMVA::MethodCFMlpANN_Utils::fg_0           = 0;
Int_t       TMVA::MethodCFMlpANN_Utils::fg_max_nVar_   = max_nVar_;
Int_t       TMVA::MethodCFMlpANN_Utils::fg_max_nNodes_ = max_nNodes_;
Int_t       TMVA::MethodCFMlpANN_Utils::fg_999         = 999;
const char* TMVA::MethodCFMlpANN_Utils::fg_MethodName  = "--- CFMlpANN                 ";

TMVA::MethodCFMlpANN_Utils::MethodCFMlpANN_Utils()
{
   // default constructor
   Int_t i(0);
   for(i=0; i<max_nVar_;++i) fVarn_1.xmin[i] = 0;
   fCost_1.ancout = 0;
   fCost_1.ieps = 0;
   fCost_1.tolcou = 0;

   for(i=0; i<max_nNodes_;++i) fDel_1.coef[i] = 0;
   for(i=0; i<max_nLayers_*max_nNodes_;++i) fDel_1.del[i] = 0;
   for(i=0; i<max_nLayers_*max_nNodes_*max_nNodes_;++i) fDel_1.delta[i] = 0;
   for(i=0; i<max_nLayers_*max_nNodes_*max_nNodes_;++i) fDel_1.delw[i] = 0;
   for(i=0; i<max_nLayers_*max_nNodes_;++i) fDel_1.delww[i] = 0;
   fDel_1.demin = 0;
   fDel_1.demax = 0;
   fDel_1.idde = 0;
   for(i=0; i<max_nLayers_;++i) fDel_1.temp[i] = 0;

   for(i=0; i<max_nNodes_;++i) fNeur_1.cut[i] = 0;
   for(i=0; i<max_nLayers_*max_nNodes_;++i) fNeur_1.deltaww[i] = 0;
   for(i=0; i<max_nLayers_;++i) fNeur_1.neuron[i] = 0;
   for(i=0; i<max_nNodes_;++i) fNeur_1.o[i] = 0;
   for(i=0; i<max_nLayers_*max_nNodes_*max_nNodes_;++i) fNeur_1.w[i] = 0;
   for(i=0; i<max_nLayers_*max_nNodes_;++i) fNeur_1.ww[i] = 0;
   for(i=0; i<max_nLayers_*max_nNodes_;++i) fNeur_1.x[i] = 0;
   for(i=0; i<max_nLayers_*max_nNodes_;++i) fNeur_1.y[i] = 0;
      
   fParam_1.eeps = 0;
   fParam_1.epsmin = 0;
   fParam_1.epsmax = 0;
   fParam_1.eta = 0;
   fParam_1.ichoi = 0;
   fParam_1.itest = 0;
   fParam_1.layerm = 0;
   fParam_1.lclass = 0;
   fParam_1.nblearn = 0;
   fParam_1.ndiv = 0;
   fParam_1.ndivis = 0;
   fParam_1.nevl = 0;
   fParam_1.nevt = 0;
   fParam_1.nunap = 0;
   fParam_1.nunilec = 0;
   fParam_1.nunishort = 0;
   fParam_1.nunisor = 0;
   fParam_1.nvar = 0;

   fVarn_1.iclass = 0;
   for(i=0; i<max_Events_;++i) fVarn_1.mclass[i] = 0;
   for(i=0; i<max_Events_;++i) fVarn_1.nclass[i] = 0;
   for(i=0; i<max_nVar_;++i) fVarn_1.xmax[i] = 0;

   fLogger = 0;
}

TMVA::MethodCFMlpANN_Utils::~MethodCFMlpANN_Utils() 
{
   // destructor
}

void TMVA::MethodCFMlpANN_Utils::Train_nn( Double_t *tin2, Double_t *tout2, Int_t *ntrain, 
                                           Int_t *ntest, Int_t *nvar2, Int_t *nlayer, 
                                           Int_t *nodes, Int_t *ncycle )
{
   // training interface - called from MethodCFMlpANN class object

   // sanity checks
   if (*ntrain + *ntest > max_Events_) {
      printf( "*** CFMlpANN_f2c: Warning in Train_nn: number of training + testing" \
              " events exceeds hardcoded maximum - reset to maximum allowed number");
      *ntrain = *ntrain*(max_Events_/(*ntrain + *ntest));
      *ntest  = *ntest *(max_Events_/(*ntrain + *ntest));
   }
   if (*nvar2 > max_nVar_) {
      printf( "*** CFMlpANN_f2c: ERROR in Train_nn: number of variables" \
              " exceeds hardcoded maximum ==> abort");
      std::exit(1);
   }
   if (*nlayer > max_nLayers_) {
      printf( "*** CFMlpANN_f2c: Warning in Train_nn: number of layers" \
              " exceeds hardcoded maximum - reset to maximum allowed number");
      *nlayer = max_nLayers_;
   }
   if (*nodes > max_nNodes_) {
      printf( "*** CFMlpANN_f2c: Warning in Train_nn: number of nodes" \
              " exceeds hardcoded maximum - reset to maximum allowed number");
      *nodes = max_nNodes_;
   }

   // create dynamic data tables (AH)
   fVarn2_1.Create( *ntrain + *ntest, *nvar2 );
   fVarn3_1.Create( *ntrain + *ntest, *nvar2 );

   Int_t imax;
   char det[20];

   Entree_new(nvar2, det, ntrain, ntest, nlayer, nodes, ncycle, (Int_t)20);
   if (fNeur_1.neuron[fParam_1.layerm - 1] == 1) {
      imax = 2;
      fParam_1.lclass = 2;
   } 
   else {
      imax = fNeur_1.neuron[fParam_1.layerm - 1] << 1;
      fParam_1.lclass = fNeur_1.neuron[fParam_1.layerm - 1];
   }
   fParam_1.nvar = fNeur_1.neuron[0];
   TestNN();
   Innit(det, tout2, tin2, (Int_t)20);

   // delete data tables
   fVarn2_1.Delete();
   fVarn3_1.Delete();
}

void TMVA::MethodCFMlpANN_Utils::Entree_new( Int_t *, char *, Int_t *ntrain, 
                                             Int_t *ntest, Int_t *numlayer, Int_t *nodes, 
                                             Int_t *numcycle, Int_t /*det_len*/)
{
   // first initialisation of ANN
   Int_t i__1;

   Int_t rewrite, i__, j, ncoef;
   Int_t ntemp, num, retrain;

   /* NTRAIN: Nb of events used during the learning */
   /* NTEST: Nb of events used for the test */
   /* TIN: Input variables */
   /* TOUT: type of the event */
 
   fCost_1.ancout = 1e30;

   /* .............. HardCoded Values .................... */
   retrain  = 0;
   rewrite  = 1000;
   for (i__ = 1; i__ <= max_nNodes_; ++i__) {
      fDel_1.coef[i__ - 1] = (Float_t)0.;
   }
   for (i__ = 1; i__ <= max_nLayers_; ++i__) {
      fDel_1.temp[i__ - 1] = (Float_t)0.;
   }
   fParam_1.layerm = *numlayer;
   if (fParam_1.layerm > max_nLayers_) {
      printf("Error: number of layers exceeds maximum: %i, %i ==> abort",
             fParam_1.layerm, max_nLayers_ );
      Arret("modification of mlpl3_param_lim.inc is needed ");
   }
   fParam_1.nevl = *ntrain;
   fParam_1.nevt = *ntest;
   fParam_1.nblearn = *numcycle;
   fVarn_1.iclass = 2;
   fParam_1.nunilec = 10;
   fParam_1.epsmin = 1e-10;
   fParam_1.epsmax = 1e-4;
   fParam_1.eta = .5;
   fCost_1.tolcou = 1e-6;
   fCost_1.ieps = 2;
   fParam_1.nunisor = 30;
   fParam_1.nunishort = 48;
   fParam_1.nunap = 40;
   
   ULog() << kINFO << "Total number of events for training: " << fParam_1.nevl << Endl;
   ULog() << kINFO << "Total number of training cycles    : " << fParam_1.nblearn << Endl;
   if (fParam_1.nevl > max_Events_) {
      printf("Error: number of learning events exceeds maximum: %i, %i ==> abort",
             fParam_1.nevl, max_Events_ );
      Arret("modification of mlpl3_param_lim.inc is needed ");
   }
   if (fParam_1.nevt > max_Events_) {
      printf("Error: number of testing events exceeds maximum: %i, %i ==> abort",
             fParam_1.nevt, max_Events_ );
      Arret("modification of mlpl3_param_lim.inc is needed ");
   }
   i__1 = fParam_1.layerm;
   for (j = 1; j <= i__1; ++j) {
      num = nodes[j-1];
      if (num < 2) {
         num = 2;
      }
      if (j == fParam_1.layerm && num != 2) {
         num = 2;
      }
      fNeur_1.neuron[j - 1] = num;
   }
   i__1 = fParam_1.layerm;
   for (j = 1; j <= i__1; ++j) {
      ULog() << kINFO << "Number of layers for neuron(" << j << "): " << fNeur_1.neuron[j - 1] << Endl;
   }
   if (fNeur_1.neuron[fParam_1.layerm - 1] != 2) {
      printf("Error: wrong number of classes at ouput layer: %i != 2 ==> abort\n",
             fNeur_1.neuron[fParam_1.layerm - 1]);
      Arret("stop");
   }
   i__1 = fNeur_1.neuron[fParam_1.layerm - 1];
   for (j = 1; j <= i__1; ++j) {
      fDel_1.coef[j - 1] = 1.;
   }
   i__1 = fParam_1.layerm;
   for (j = 1; j <= i__1; ++j) {
      fDel_1.temp[j - 1] = 1.;
   }
   fParam_1.ichoi = retrain;
   fParam_1.ndivis = rewrite;
   fDel_1.idde = 1;
   if (! (fParam_1.ichoi == 0 || fParam_1.ichoi == 1)) {
      printf( "Big troubles !!! \n" );
      Arret("new training or continued one !");
   }
   if (fParam_1.ichoi == 0) {
      ULog() << kINFO << "New training will be performed" << Endl;
   }
   else {
      printf("%s: New training will be continued from a weight file\n", fg_MethodName);
   }
   ncoef = 0;
   ntemp = 0;
   for (i__ = 1; i__ <= max_nNodes_; ++i__) {
      if (fDel_1.coef[i__ - 1] != (Float_t)0.) {
         ++ncoef;
      }
   }
   for (i__ = 1; i__ <= max_nLayers_; ++i__) {
      if (fDel_1.temp[i__ - 1] != (Float_t)0.) {
         ++ntemp;
      }
   }
   if (ncoef != fNeur_1.neuron[fParam_1.layerm - 1]) {
      Arret(" entree error code 1 : need to reported");
   }
   if (ntemp != fParam_1.layerm) {
      Arret("entree error code 2 : need to reported");
   }
}

#define w_ref(a_1,a_2,a_3) fNeur_1.w[((a_3)*max_nNodes_ + (a_2))*max_nLayers_ + a_1 - 187]
#define ww_ref(a_1,a_2) fNeur_1.ww[(a_2)*max_nLayers_ + a_1 - 7]

void TMVA::MethodCFMlpANN_Utils::Wini()
{
   // [smart comments to be added]
   Int_t i__1, i__2, i__3;
   Int_t i__, j;
   Int_t layer;

   i__1 = fParam_1.layerm;
   for (layer = 2; layer <= i__1; ++layer) {
      i__2 = fNeur_1.neuron[layer - 2];
      for (i__ = 1; i__ <= i__2; ++i__) {
         i__3 = fNeur_1.neuron[layer - 1];
         for (j = 1; j <= i__3; ++j) {
            w_ref(layer, j, i__) = (Sen3a() * 2. - 1.) * .2;
            ww_ref(layer, j) = (Sen3a() * 2. - 1.) * .2;
         }
      }
   }
}

#undef ww_ref
#undef w_ref

#define xeev_ref(a_1,a_2) fVarn2_1(a_1,a_2)
#define w_ref(a_1,a_2,a_3) fNeur_1.w[((a_3)*max_nNodes_ + (a_2))*max_nLayers_ + a_1 - 187]
#define x_ref(a_1,a_2) fNeur_1.x[(a_2)*max_nLayers_ + a_1 - 7]
#define y_ref(a_1,a_2) fNeur_1.y[(a_2)*max_nLayers_ + a_1 - 7]
#define ww_ref(a_1,a_2) fNeur_1.ww[(a_2)*max_nLayers_ + a_1 - 7]

void TMVA::MethodCFMlpANN_Utils::En_avant(Int_t *ievent)
{
   // [smart comments to be added]
   Int_t i__1, i__2, i__3;

   Double_t f;
   Int_t i__, j;
   Int_t layer;
   
   i__1 = fNeur_1.neuron[0];
   for (i__ = 1; i__ <= i__1; ++i__) {
      y_ref(1, i__) = xeev_ref(*ievent, i__);
   }
   i__1 = fParam_1.layerm - 1;
   for (layer = 1; layer <= i__1; ++layer) {
      i__2 = fNeur_1.neuron[layer];
      for (j = 1; j <= i__2; ++j) {
         x_ref(layer + 1, j) = 0.;
         i__3 = fNeur_1.neuron[layer - 1];
         for (i__ = 1; i__ <= i__3; ++i__) {
            x_ref(layer + 1, j) = ( x_ref(layer + 1, j) + y_ref(layer, i__) 
                                    * w_ref(layer + 1, j, i__) );
         }
         x_ref(layer + 1, j) = x_ref(layer + 1, j) + ww_ref(layer + 1, j);
         i__3 = layer + 1;
         Foncf(&i__3, &x_ref(layer + 1, j), &f);
         y_ref(layer + 1, j) = f;
      }
   }
} 

#undef ww_ref
#undef y_ref
#undef x_ref
#undef w_ref
#undef xeev_ref

#define xeev_ref(a_1,a_2) fVarn2_1(a_1,a_2)

void TMVA::MethodCFMlpANN_Utils::Leclearn( Int_t *ktest, Double_t *tout2, Double_t *tin2 )
{
   // [smart comments to be added]
   Int_t i__1, i__2;

   Int_t i__, j, k, l;
   Int_t nocla[max_nNodes_], ikend;
   Double_t xpg[max_nVar_];

   *ktest = 0;
   i__1 = fParam_1.lclass;
   for (k = 1; k <= i__1; ++k) {
      nocla[k - 1] = 0;
   }
   i__1 = fParam_1.nvar;
   for (i__ = 1; i__ <= i__1; ++i__) {
      fVarn_1.xmin[i__ - 1] = 1e30;
      fVarn_1.xmax[i__ - 1] = -fVarn_1.xmin[i__ - 1];
   }
   i__1 = fParam_1.nevl;
   for (i__ = 1; i__ <= i__1; ++i__) {
      DataInterface(tout2, tin2, &fg_100, &fg_0, &fParam_1.nevl, &fParam_1.nvar, 
                    xpg, &fVarn_1.nclass[i__ - 1], &ikend);
      if (ikend == -1) {
         break;
      }

      CollectVar(&fParam_1.nvar, &fVarn_1.nclass[i__ - 1], xpg);

      i__2 = fParam_1.nvar;
      for (j = 1; j <= i__2; ++j) {        
         xeev_ref(i__, j) = xpg[j - 1];
      }
      if (fVarn_1.iclass == 1) {
         i__2 = fParam_1.lclass;
         for (k = 1; k <= i__2; ++k) {
            if (fVarn_1.nclass[i__ - 1] == k) {
               ++nocla[k - 1];
            }
         }
      }
      i__2 = fParam_1.nvar;
      for (k = 1; k <= i__2; ++k) {
         if (xeev_ref(i__, k) < fVarn_1.xmin[k - 1]) {
            fVarn_1.xmin[k - 1] = xeev_ref(i__, k);
         }
         if (xeev_ref(i__, k) > fVarn_1.xmax[k - 1]) {
            fVarn_1.xmax[k - 1] = xeev_ref(i__, k);
         }
      }
   }

   if (fVarn_1.iclass == 1) {
      i__2 = fParam_1.lclass;
      for (k = 1; k <= i__2; ++k) {
         i__1 = fParam_1.lclass;
         for (l = 1; l <= i__1; ++l) {
            if (nocla[k - 1] != nocla[l - 1]) {
               *ktest = 1;
            }
         }
      }
   }
   i__1 = fParam_1.nevl;
   for (i__ = 1; i__ <= i__1; ++i__) {
      i__2 = fParam_1.nvar;
      for (l = 1; l <= i__2; ++l) {
         if (fVarn_1.xmax[l - 1] == (Float_t)0. && fVarn_1.xmin[l - 1] == (
                                                                           Float_t)0.) {
            xeev_ref(i__, l) = (Float_t)0.;
         } 
         else {
            xeev_ref(i__, l) = xeev_ref(i__, l) - (fVarn_1.xmax[l - 1] + 
                                                   fVarn_1.xmin[l - 1]) / 2.;
            xeev_ref(i__, l) = xeev_ref(i__, l) / ((fVarn_1.xmax[l - 1] - 
                                                    fVarn_1.xmin[l - 1]) / 2.);
         }
      }
   }
}

#undef xeev_ref

#define delw_ref(a_1,a_2,a_3) fDel_1.delw[((a_3)*max_nNodes_ + (a_2))*max_nLayers_ + a_1 - 187]
#define w_ref(a_1,a_2,a_3) fNeur_1.w[((a_3)*max_nNodes_ + (a_2))*max_nLayers_ + a_1 - 187]
#define x_ref(a_1,a_2) fNeur_1.x[(a_2)*max_nLayers_ + a_1 - 7]
#define y_ref(a_1,a_2) fNeur_1.y[(a_2)*max_nLayers_ + a_1 - 7]
#define delta_ref(a_1,a_2,a_3) fDel_1.delta[((a_3)*max_nNodes_ + (a_2))*max_nLayers_ + a_1 - 187]
#define delww_ref(a_1,a_2) fDel_1.delww[(a_2)*max_nLayers_ + a_1 - 7]
#define ww_ref(a_1,a_2) fNeur_1.ww[(a_2)*max_nLayers_ + a_1 - 7]
#define del_ref(a_1,a_2) fDel_1.del[(a_2)*max_nLayers_ + a_1 - 7]
#define deltaww_ref(a_1,a_2) fNeur_1.deltaww[(a_2)*max_nLayers_ + a_1 - 7]

void TMVA::MethodCFMlpANN_Utils::En_arriere( Int_t *ievent )
{
   // [smart comments to be added]
   Int_t i__1, i__2, i__3;

   Double_t f;
   Int_t i__, j, k, l;
   Double_t df, uu;

   i__1 = fNeur_1.neuron[fParam_1.layerm - 1];
   for (i__ = 1; i__ <= i__1; ++i__) {
      if (fVarn_1.nclass[*ievent - 1] == i__) {
         fNeur_1.o[i__ - 1] = 1.;
      } 
      else {
         fNeur_1.o[i__ - 1] = -1.;
      }
   }
   l = fParam_1.layerm;
   i__1 = fNeur_1.neuron[l - 1];
   for (i__ = 1; i__ <= i__1; ++i__) {
      f = y_ref(l, i__);
      df = (f + 1.) * (1. - f) / (fDel_1.temp[l - 1] * 2.);
      del_ref(l, i__) = df * (fNeur_1.o[i__ - 1] - y_ref(l, i__)) * 
         fDel_1.coef[i__ - 1];
      delww_ref(l, i__) = fParam_1.eeps * del_ref(l, i__);
      i__2 = fNeur_1.neuron[l - 2];
      for (j = 1; j <= i__2; ++j) {
         delw_ref(l, i__, j) = fParam_1.eeps * del_ref(l, i__) * y_ref(l - 
                                                                       1, j);
         /* L20: */
      }
   }
   for (l = fParam_1.layerm - 1; l >= 2; --l) {
      i__2 = fNeur_1.neuron[l - 1];
      for (i__ = 1; i__ <= i__2; ++i__) {
         uu = 0.;
         i__1 = fNeur_1.neuron[l];
         for (k = 1; k <= i__1; ++k) {
            uu += w_ref(l + 1, k, i__) * del_ref(l + 1, k);
         }
         Foncf(&l, &x_ref(l, i__), &f);
         df = (f + 1.) * (1. - f) / (fDel_1.temp[l - 1] * 2.);
         del_ref(l, i__) = df * uu;
         delww_ref(l, i__) = fParam_1.eeps * del_ref(l, i__);
         i__1 = fNeur_1.neuron[l - 2];
         for (j = 1; j <= i__1; ++j) {
            delw_ref(l, i__, j) = fParam_1.eeps * del_ref(l, i__) * y_ref(
                                                                          l - 1, j);
         }
      }
   }
   i__1 = fParam_1.layerm;
   for (l = 2; l <= i__1; ++l) {
      i__2 = fNeur_1.neuron[l - 1];
      for (i__ = 1; i__ <= i__2; ++i__) {
         deltaww_ref(l, i__) = delww_ref(l, i__) + fParam_1.eta * 
            deltaww_ref(l, i__);
         ww_ref(l, i__) = ww_ref(l, i__) + deltaww_ref(l, i__);
         i__3 = fNeur_1.neuron[l - 2];
         for (j = 1; j <= i__3; ++j) {
            delta_ref(l, i__, j) = delw_ref(l, i__, j) + fParam_1.eta * 
               delta_ref(l, i__, j);
            w_ref(l, i__, j) = w_ref(l, i__, j) + delta_ref(l, i__, j);
         }
      }
   }
}

#undef deltaww_ref
#undef del_ref
#undef ww_ref
#undef delww_ref
#undef delta_ref
#undef y_ref
#undef x_ref
#undef w_ref
#undef delw_ref

#define w_ref(a_1,a_2,a_3) fNeur_1.w[((a_3)*max_nNodes_ + (a_2))*max_nLayers_ + a_1 - 187]
#define ww_ref(a_1,a_2) fNeur_1.ww[(a_2)*max_nLayers_ + a_1 - 7]

void TMVA::MethodCFMlpANN_Utils::Out( Int_t *iii, Int_t *maxcycle )
{
   // write weights to file

   if (*iii == *maxcycle) {
      // now in MethodCFMlpANN.cxx
   }
}

#undef ww_ref
#undef w_ref

#define delta_ref(a_1,a_2,a_3) fDel_1.delta[((a_3)*max_nNodes_ + (a_2))*max_nLayers_ + a_1 - 187]
#define deltaww_ref(a_1,a_2) fNeur_1.deltaww[(a_2)*max_nLayers_ + a_1 - 7]

void TMVA::MethodCFMlpANN_Utils::Innit( char *det, Double_t *tout2, Double_t *tin2, Int_t )
{
   // Initialization
   Int_t i__1, i__2, i__3;

   Int_t i__, j;
   Int_t nevod, layer, ktest, i1, nrest;
   Int_t ievent(0);
   Int_t kkk;
   Double_t xxx, yyy;

   Leclearn(&ktest, tout2, tin2);
   Lecev2(&ktest, tout2, tin2);
   if (ktest == 1) {
      printf( " .... strange to be here (1) ... \n");
      std::exit(1);
   }
   i__1 = fParam_1.layerm - 1;
   for (layer = 1; layer <= i__1; ++layer) {
      i__2 = fNeur_1.neuron[layer];
      for (j = 1; j <= i__2; ++j) {
         deltaww_ref(layer + 1, j) = 0.;
         i__3 = fNeur_1.neuron[layer - 1];
         for (i__ = 1; i__ <= i__3; ++i__) {
            delta_ref(layer + 1, j, i__) = 0.;
         }
      }
   }
   if (fParam_1.ichoi == 1) {
      Inl();
   } 
   else {
      Wini();
   }
   kkk = 0;
   i__3 = fParam_1.nblearn;
   Timer timer( i__3, "CFMlpANN" ); 
   Int_t num = i__3/100;

   for (i1 = 1; i1 <= i__3; ++i1) {

      if ( ( num>0 && (i1-1)%num == 0) || (i1 == i__3) ) timer.DrawProgressBar( i1-1 );

      i__2 = fParam_1.nevl;
      for (i__ = 1; i__ <= i__2; ++i__) {
         ++kkk;
         if (fCost_1.ieps == 2) {
            fParam_1.eeps = Fdecroi(&kkk);
         }
         if (fCost_1.ieps == 1) {
            fParam_1.eeps = fParam_1.epsmin;
         }
         Bool_t doCont = kTRUE;
         if (fVarn_1.iclass == 2) {
            ievent = (Int_t) ((Double_t) fParam_1.nevl * Sen3a());
            if (ievent == 0) {
               doCont = kFALSE;
            }
         }
         if (doCont) {
            if (fVarn_1.iclass == 1) {
               nevod = fParam_1.nevl / fParam_1.lclass;
               nrest = i__ % fParam_1.lclass;
               fParam_1.ndiv = i__ / fParam_1.lclass;
               if (nrest != 0) {
                  ievent = fParam_1.ndiv + 1 + (fParam_1.lclass - nrest) * 
                     nevod;
               } 
               else {
                  ievent = fParam_1.ndiv;
               }
            }
            En_avant(&ievent);
            En_arriere(&ievent);
         }
      }
      yyy = 0.;
      if (i1 % fParam_1.ndivis == 0 || i1 == 1 || i1 == fParam_1.nblearn) {
         Cout(&i1, &xxx);
         Cout2(&i1, &yyy);
         GraphNN(&i1, &xxx, &yyy, det, (Int_t)20);
         Out(&i1, &fParam_1.nblearn);
      }
      if (xxx < fCost_1.tolcou) {
         GraphNN(&fParam_1.nblearn, &xxx, &yyy, det, (Int_t)20);
         Out(&fParam_1.nblearn, &fParam_1.nblearn);
         break;
      }
   }
}

#undef deltaww_ref
#undef delta_ref

void TMVA::MethodCFMlpANN_Utils::TestNN()
{
   // [smart comments to be added]
   Int_t i__1;

   Int_t i__;
   Int_t ktest;

   ktest = 0;
   if (fParam_1.layerm > max_nLayers_) {
      ktest = 1;
      printf("Error: number of layers exceeds maximum: %i, %i ==> abort", 
             fParam_1.layerm, max_nLayers_ );
      Arret("modification of mlpl3_param_lim.inc is needed ");
   }
   if (fParam_1.nevl > max_Events_) {
      ktest = 1;
      printf("Error: number of training events exceeds maximum: %i, %i ==> abort", 
             fParam_1.nevl, max_Events_ );
      Arret("modification of mlpl3_param_lim.inc is needed ");
   }
   if (fParam_1.nevt > max_Events_) {
      printf("Error: number of testing events exceeds maximum: %i, %i ==> abort", 
             fParam_1.nevt, max_Events_ );
      Arret("modification of mlpl3_param_lim.inc is needed ");
   }
   if (fParam_1.lclass < fNeur_1.neuron[fParam_1.layerm - 1]) {
      ktest = 1;
      printf("Error: wrong number of classes at ouput layer: %i != %i ==> abort\n",
             fNeur_1.neuron[fParam_1.layerm - 1], fParam_1.lclass);
      Arret("problem needs to reported ");
   }
   if (fParam_1.nvar > max_nVar_) {
      ktest = 1;
      printf("Error: number of variables exceeds maximum: %i, %i ==> abort", 
             fParam_1.nvar, fg_max_nVar_ );
      Arret("modification of mlpl3_param_lim.inc is needed");
   }
   i__1 = fParam_1.layerm;
   for (i__ = 1; i__ <= i__1; ++i__) {
      if (fNeur_1.neuron[i__ - 1] > max_nNodes_) {
         ktest = 1;
         printf("Error: number of neurons at layer exceeds maximum: %i, %i ==> abort", 
                i__, fg_max_nNodes_ );
      }
   }
   if (ktest == 1) {
      printf( " .... strange to be here (2) ... \n");
      std::exit(1);
   }
}

#define y_ref(a_1,a_2) fNeur_1.y[(a_2)*max_nLayers_ + a_1 - 7]

void TMVA::MethodCFMlpANN_Utils::Cout( Int_t * /*i1*/, Double_t *xxx )
{
   // [smart comments to be added]
   Int_t i__1, i__2;
   Double_t d__1;
   
   Double_t c__;
   Int_t i__, j;
   
   c__ = 0.;
   i__1 = fParam_1.nevl;
   for (i__ = 1; i__ <= i__1; ++i__) {
      En_avant(&i__);
      i__2 = fNeur_1.neuron[fParam_1.layerm - 1];
      for (j = 1; j <= i__2; ++j) {
         if (fVarn_1.nclass[i__ - 1] == j) {
            fNeur_1.o[j - 1] = 1.;
         } 
         else {
            fNeur_1.o[j - 1] = -1.;
         }
         // Computing 2nd power 
         d__1 = y_ref(fParam_1.layerm, j) - fNeur_1.o[j - 1];
         c__ += fDel_1.coef[j - 1] * (d__1 * d__1);
      }
   }
   c__ /= (Double_t) (fParam_1.nevl * fParam_1.lclass) * 2.;
   *xxx = c__;
   fCost_1.ancout = c__;
}

#undef y_ref

#define w_ref(a_1,a_2,a_3) fNeur_1.w[((a_3)*max_nNodes_ + (a_2))*max_nLayers_ + a_1 - 187]
#define ww_ref(a_1,a_2) fNeur_1.ww[(a_2)*max_nLayers_ + a_1 - 7]

void TMVA::MethodCFMlpANN_Utils::Inl()
{
   // [smart comments to be added]
   Int_t i__1, i__2, i__3;

   Int_t jmin, jmax, k, layer, kk, nq, nr;

   i__1 = fParam_1.nvar;
   i__1 = fParam_1.layerm;
   i__1 = fParam_1.layerm - 1;
   for (layer = 1; layer <= i__1; ++layer) {
      nq = fNeur_1.neuron[layer] / 10;
      nr = fNeur_1.neuron[layer] - nq * 10;
      if (nr == 0) {
         kk = nq;
      } 
      else {
         kk = nq + 1;
      }
      i__2 = kk;
      for (k = 1; k <= i__2; ++k) {
         jmin = k * 10 - 9;
         jmax = k * 10;
         if (fNeur_1.neuron[layer] < jmax) {
            jmax = fNeur_1.neuron[layer];
         }
         i__3 = fNeur_1.neuron[layer - 1];
      }
   }
}

#undef ww_ref
#undef w_ref

Double_t TMVA::MethodCFMlpANN_Utils::Fdecroi( Int_t *i__ )
{
   // [smart comments to be added]
   Double_t ret_val;
   
   Double_t aaa, bbb;
   
   aaa = (fParam_1.epsmin - fParam_1.epsmax) / (Double_t) (fParam_1.nblearn * 
                                                           fParam_1.nevl - 1);
   bbb = fParam_1.epsmax - aaa;
   ret_val = aaa * (Double_t) (*i__) + bbb;
   return ret_val;
}

#define y_ref(a_1,a_2) fNeur_1.y[(a_2)*max_nLayers_ + a_1 - 7]

void TMVA::MethodCFMlpANN_Utils::GraphNN( Int_t *ilearn, Double_t * /*xxx*/, 
                                          Double_t * /*yyy*/, char * /*det*/, Int_t  /*det_len*/ )
{
   // [smart comments to be added]
   Int_t i__1, i__2;
   
   Double_t xmok[max_nNodes_];
   Float_t xpaw;
   Double_t xmko[max_nNodes_];
   Int_t i__, j;
   Int_t ix;
   Int_t jjj;
   Float_t vbn[10];
   Int_t nko[max_nNodes_], nok[max_nNodes_];

   for (i__ = 1; i__ <= 10; ++i__) {
      vbn[i__ - 1] = (Float_t)0.;
   }
   if (*ilearn == 1) {
      // AH: removed output 
   }
   i__1 = fNeur_1.neuron[fParam_1.layerm - 1];
   for (i__ = 1; i__ <= i__1; ++i__) {
      nok[i__ - 1] = 0;
      nko[i__ - 1] = 0;
      xmok[i__ - 1] = 0.;
      xmko[i__ - 1] = 0.;
   }
   i__1 = fParam_1.nevl;
   for (i__ = 1; i__ <= i__1; ++i__) {
      En_avant(&i__);
      i__2 = fNeur_1.neuron[fParam_1.layerm - 1];
      for (j = 1; j <= i__2; ++j) {
         xpaw = (Float_t) y_ref(fParam_1.layerm, j);
         if (fVarn_1.nclass[i__ - 1] == j) {
            ++nok[j - 1];
            xmok[j - 1] += y_ref(fParam_1.layerm, j);
         } 
         else {
            ++nko[j - 1];
            xmko[j - 1] += y_ref(fParam_1.layerm, j);
            jjj = j + fNeur_1.neuron[fParam_1.layerm - 1];
         }
         if (j <= 9) {
            vbn[j - 1] = xpaw;
         }
      }
      vbn[9] = (Float_t) fVarn_1.nclass[i__ - 1];
   }
   i__1 = fNeur_1.neuron[fParam_1.layerm - 1];
   for (j = 1; j <= i__1; ++j) {
      xmok[j - 1] /= (Double_t) nok[j - 1];
      xmko[j - 1] /= (Double_t) nko[j - 1];
      fNeur_1.cut[j - 1] = (xmok[j - 1] + xmko[j - 1]) / 2.;
   }
   ix = fNeur_1.neuron[fParam_1.layerm - 1];
   i__1 = ix;
}

#undef y_ref

Double_t TMVA::MethodCFMlpANN_Utils::Sen3a( void )
{
   // [smart comments to be added]

   // Initialized data
   Int_t    m12 = 4096;
   Double_t f1  = 2.44140625e-4;
   Double_t f2  = 5.96046448e-8;
   Double_t f3  = 1.45519152e-11;
   Int_t    j1  = 3823;
   Int_t    j2  = 4006;
   Int_t    j3  = 2903;
   static Int_t fg_i1 = 3823;
   static Int_t fg_i2 = 4006;
   static Int_t fg_i3 = 2903;

   Double_t ret_val;
   Int_t    k3, l3, k2, l2, k1, l1;

   // reference: /k.d.senne/j. stochastics/ vol 1,no 3 (1974),pp.215-38 
   k3 = fg_i3 * j3;
   l3 = k3 / m12;
   k2 = fg_i2 * j3 + fg_i3 * j2 + l3;
   l2 = k2 / m12;
   k1 = fg_i1 * j3 + fg_i2 * j2 + fg_i3 * j1 + l2;
   l1 = k1 / m12;
   fg_i1 = k1 - l1 * m12;
   fg_i2 = k2 - l2 * m12;
   fg_i3 = k3 - l3 * m12;
   ret_val = f1 * (Double_t) fg_i1 + f2 * (Float_t) fg_i2 + f3 * (Double_t) fg_i3;

   return ret_val;
} 

void TMVA::MethodCFMlpANN_Utils::Foncf( Int_t *i__, Double_t *u, Double_t *f )
{
   // [needs to be checked]
   Double_t yy;

   if (*u / fDel_1.temp[*i__ - 1] > 170.) {
      *f = .99999999989999999;
   } 
   else if (*u / fDel_1.temp[*i__ - 1] < -170.) {
      *f = -.99999999989999999;
   } 
   else {
      yy = TMath::Exp(-(*u) / fDel_1.temp[*i__ - 1]);
      *f = (1. - yy) / (yy + 1.);
   }
}

#undef w_ref

#define y_ref(a_1,a_2) fNeur_1.y[(a_2)*max_nLayers_ + a_1 - 7]

void TMVA::MethodCFMlpANN_Utils::Cout2( Int_t * /*i1*/, Double_t *yyy )
{
   // [smart comments to be added]
   Int_t i__1, i__2;
   Double_t d__1;

   Double_t c__;
   Int_t i__, j;

   c__ = 0.;
   i__1 = fParam_1.nevt;
   for (i__ = 1; i__ <= i__1; ++i__) {
      En_avant2(&i__);
      i__2 = fNeur_1.neuron[fParam_1.layerm - 1];
      for (j = 1; j <= i__2; ++j) {
         if (fVarn_1.mclass[i__ - 1] == j) {
            fNeur_1.o[j - 1] = 1.;
         } 
         else {
            fNeur_1.o[j - 1] = -1.;
         }
         /* Computing 2nd power */
         d__1 = y_ref(fParam_1.layerm, j) - fNeur_1.o[j - 1];
         c__ += fDel_1.coef[j - 1] * (d__1 * d__1);
      }
   }
   c__ /= (Double_t) (fParam_1.nevt * fParam_1.lclass) * 2.;
   *yyy = c__;
}

#undef y_ref

#define xx_ref(a_1,a_2) fVarn3_1(a_1,a_2)

void TMVA::MethodCFMlpANN_Utils::Lecev2( Int_t *ktest, Double_t *tout2, Double_t *tin2 )
{
   // [smart comments to be added]
   Int_t i__1, i__2;

   Int_t i__, j, k, l, mocla[max_nNodes_], ikend;
   Double_t xpg[max_nVar_];

   /* NTRAIN: Nb of events used during the learning */
   /* NTEST: Nb of events used for the test */
   /* TIN: Input variables */
   /* TOUT: type of the event */

   *ktest = 0;
   i__1 = fParam_1.lclass;
   for (k = 1; k <= i__1; ++k) {
      mocla[k - 1] = 0;
   }
   i__1 = fParam_1.nevt;
   for (i__ = 1; i__ <= i__1; ++i__) {
      DataInterface(tout2, tin2, &fg_999, &fg_0, &fParam_1.nevt, &fParam_1.nvar, 
                    xpg, &fVarn_1.mclass[i__ - 1], &ikend);

      if (ikend == -1) {
         break;
      }

      i__2 = fParam_1.nvar;
      for (j = 1; j <= i__2; ++j) {
         xx_ref(i__, j) = xpg[j - 1];
      }
   }
 
   i__1 = fParam_1.nevt;
   for (i__ = 1; i__ <= i__1; ++i__) {
      i__2 = fParam_1.nvar;
      for (l = 1; l <= i__2; ++l) {
         if (fVarn_1.xmax[l - 1] == (Float_t)0. && fVarn_1.xmin[l - 1] == (
                                                                           Float_t)0.) {
            xx_ref(i__, l) = (Float_t)0.;
         } 
         else {
            xx_ref(i__, l) = xx_ref(i__, l) - (fVarn_1.xmax[l - 1] + 
                                               fVarn_1.xmin[l - 1]) / 2.;
            xx_ref(i__, l) = xx_ref(i__, l) / ((fVarn_1.xmax[l - 1] - 
                                                fVarn_1.xmin[l - 1]) / 2.);
         }
      }
   }
} 

#undef xx_ref

#define w_ref(a_1,a_2,a_3) fNeur_1.w[((a_3)*max_nNodes_ + (a_2))*max_nLayers_ + a_1 - 187]
#define x_ref(a_1,a_2) fNeur_1.x[(a_2)*max_nLayers_ + a_1 - 7]
#define y_ref(a_1,a_2) fNeur_1.y[(a_2)*max_nLayers_ + a_1 - 7]
#define ww_ref(a_1,a_2) fNeur_1.ww[(a_2)*max_nLayers_ + a_1 - 7]
#define xx_ref(a_1,a_2) fVarn3_1(a_1,a_2)

void TMVA::MethodCFMlpANN_Utils::En_avant2( Int_t *ievent )
{
   // [smart comments to be added]
   Int_t i__1, i__2, i__3;

   Double_t f;
   Int_t i__, j;
   Int_t layer;

   i__1 = fNeur_1.neuron[0];
   for (i__ = 1; i__ <= i__1; ++i__) {
      y_ref(1, i__) = xx_ref(*ievent, i__);
   }
   i__1 = fParam_1.layerm - 1;
   for (layer = 1; layer <= i__1; ++layer) {
      i__2 = fNeur_1.neuron[layer];
      for (j = 1; j <= i__2; ++j) {
         x_ref(layer + 1, j) = 0.;
         i__3 = fNeur_1.neuron[layer - 1];
         for (i__ = 1; i__ <= i__3; ++i__) {
            x_ref(layer + 1, j) = x_ref(layer + 1, j) + y_ref(layer, i__) 
               * w_ref(layer + 1, j, i__);
         }
         x_ref(layer + 1, j) = x_ref(layer + 1, j) + ww_ref(layer + 1, j);
         i__3 = layer + 1;
         Foncf(&i__3, &x_ref(layer + 1, j), &f);
         y_ref(layer + 1, j) = f;
         /* L2: */
      }
   }
}

#undef xx_ref
#undef ww_ref
#undef y_ref
#undef x_ref
#undef w_ref

void TMVA::MethodCFMlpANN_Utils::Arret( const char* mot )
{
   // fatal error occurred: stop execution
   printf("%s: %s",fg_MethodName, mot);
   std::exit(1);
}

void TMVA::MethodCFMlpANN_Utils::CollectVar( Int_t *nvar, Int_t *class__, Double_t *xpg )
{
   // [smart comments to be added]
   Int_t i__1;
   
   Int_t i__;
   Float_t x[201];

   // Parameter adjustments
   --xpg;

   for (i__ = 1; i__ <= 201; ++i__) {
      x[i__ - 1] = 0.0;
   }
   x[0] = (Float_t) (*class__);
   i__1 = *nvar;
   for (i__ = 1; i__ <= i__1; ++i__) {
      x[i__] = (Float_t) xpg[i__];
   }
}
