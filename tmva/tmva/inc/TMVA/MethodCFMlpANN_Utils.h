// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodCFMlpANN_utils                                                  *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Reference for the original FORTRAN version "mlpl3.F":                          *
 *      Authors  : J. Proriol and contributions from ALEPH-Clermont-Fd            *
 *                 Team members                                                   *
 *      Copyright: Laboratoire Physique Corpusculaire                             *
 *                 Universite de Blaise Pascal, IN2P3/CNRS                        *
 * Description:                                                                   *
 *      Utility routine, obtained via f2c from original mlpl3.F FORTRAN routine   *
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
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodCFMlpANN_Utils
#define ROOT_TMVA_MethodCFMlpANN_Utils

#include "TMVA/MethodCFMlpANN_def.h"
#include "TMVA/MsgLogger.h"

#include "Rtypes.h"

#include <cstdlib>
//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodCFMlpANN_Utils                                                 //
//                                                                      //
// Implementation of Clermond-Ferrand artificial neural network         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

namespace TMVA {

   class MethodCFMlpANN_Utils {

   public:

      MethodCFMlpANN_Utils();
      virtual ~MethodCFMlpANN_Utils();

   protected:

      void Train_nn( Double_t *tin2, Double_t *tout2, Int_t *ntrain,
                     Int_t *ntest, Int_t *nvar2, Int_t *nlayer,
                     Int_t *nodes, Int_t *ncycle );

      void Entree_new( Int_t *, char *, Int_t *ntrain, Int_t *ntest,
                       Int_t *numlayer, Int_t *nodes, Int_t *numcycle,
                       Int_t );

      virtual Int_t DataInterface( Double_t*, Double_t*, Int_t*, Int_t*, Int_t*, Int_t*,
                                   Double_t*, Int_t*, Int_t* ) = 0;

      Double_t Fdecroi(Int_t *i__);
      Double_t Sen3a(void);

      void  Wini      ();
      void  En_avant  (Int_t *ievent);
      void  En_avant2 (Int_t *ievent);
      void  En_arriere(Int_t *ievent);
      void  Leclearn  (Int_t *ktest, Double_t *tout2, Double_t *tin2);
      void  Out       (Int_t *iii, Int_t *maxcycle);
      void  Cout      (Int_t *, Double_t *xxx);
      void  Innit     (char *det, Double_t *tout2, Double_t *tin2, Int_t );
      void  TestNN    ();
      void  Inl       ();
      void  GraphNN   (Int_t *ilearn, Double_t *, Double_t *, char *, Int_t);
      void  Foncf     (Int_t *i__, Double_t *u, Double_t *f);
      void  Cout2     (Int_t * /*i1*/, Double_t *yyy);
      void  Lecev2    (Int_t *ktest, Double_t *tout2, Double_t *tin2);
      void  Arret     (const char* mot );
      void  CollectVar(Int_t *nvar, Int_t *class__, Double_t *xpg);

   protected:

      Int_t             fg_100;          // constant
      Int_t             fg_0;            // constant
      static const Int_t       fg_max_nVar_;    // static maximum number of input variables
      static const Int_t       fg_max_nNodes_;  // maximum number of nodes per variable
      Int_t             fg_999;          // constant
      static const char* const fg_MethodName;   // method name for print

      Double_t W_ref(const Double_t wNN[], Int_t a_1, Int_t a_2, Int_t a_3) const {
         return wNN [(a_3*max_nNodes_ + a_2)*max_nLayers_ + a_1 - 187];
      }
      Double_t& W_ref(Double_t wNN[], Int_t a_1, Int_t a_2, Int_t a_3) {
         return wNN [((a_3)*max_nNodes_ + (a_2))*max_nLayers_ + a_1 - 187];
      }

      Double_t Ww_ref(const Double_t wwNN[], Int_t a_1,Int_t a_2) const {
         return wwNN[(a_2)*max_nLayers_ + a_1 - 7];
      }
      Double_t& Ww_ref(Double_t wwNN[], Int_t a_1,Int_t a_2) {
         return wwNN[(a_2)*max_nLayers_ + a_1 - 7];
      }

      // ANN training parameters
      struct {
         Double_t epsmin, epsmax, eeps, eta;
         Int_t layerm, lclass, nevl, nblearn, nunilec, nunisor, nunishort, nunap;
         Int_t nvar, itest, ndiv, ichoi, ndivis, nevt;
      } fParam_1;

      // ANN training results
      struct {
         Double_t xmax[max_nVar_], xmin[max_nVar_];
         Int_t nclass[max_Events_], mclass[max_Events_], iclass;
      } fVarn_1;

      // dynamic data table
      class VARn2 {
      public:
         VARn2() : fNevt(0), fNvar(0) {
            fxx = nullptr;
         }
         ~VARn2() {
            Delete();
         }
         void Create( Int_t nevt, Int_t nvar ) {
            fNevt = nevt+1; fNvar = nvar+1; // fortran array style 1...N
            fxx = new Double_t*[fNevt];
            for (Int_t i=0; i<fNevt; i++) fxx[i] = new Double_t[fNvar];
         }
         Double_t operator=( Double_t val ) { return val; }
         Double_t &operator()( Int_t ievt, Int_t ivar ) const {
            if (fxx && ievt < fNevt && ivar < fNvar) return fxx[ievt][ivar];
            else {
               printf( "*** ERROR in varn3_(): fxx is zero pointer ==> abort ***\n") ;
               std::exit(1);
               return fxx[0][0];
            }
         }
         void Delete( void ) {
            if (fxx) for (Int_t i=0; i<fNevt; i++) if (fxx[i]) delete [] fxx[i];
            delete[] fxx;
            fxx=nullptr;
         }

         Double_t** fxx;
         Int_t fNevt;
         Int_t fNvar;
      } fVarn2_1, fVarn3_1;

      // ANN weights
      struct {
         Double_t x[max_nLayers_*max_nNodes_];
         Double_t y[max_nLayers_*max_nNodes_];
         Double_t o[max_nNodes_];
         Double_t w[max_nLayers_*max_nNodes_*max_nNodes_];
         Double_t ww[max_nLayers_*max_nNodes_];
         Double_t cut[max_nNodes_];
         Double_t deltaww[max_nLayers_*max_nNodes_];
         Int_t neuron[max_nLayers_];
      } fNeur_1;

      // ANN weights
      struct {
         Double_t coef[max_nNodes_], temp[max_nLayers_], demin, demax;
         Double_t del[max_nLayers_*max_nNodes_];
         Double_t delw[max_nLayers_*max_nNodes_*max_nNodes_];
         Double_t delta[max_nLayers_*max_nNodes_*max_nNodes_];
         Double_t delww[max_nLayers_*max_nNodes_];
         Int_t idde;
      } fDel_1;

      // flags and stuff (don't ask me...)
      struct {
         Double_t ancout, tolcou;
         Int_t ieps;
      } fCost_1;

      void SetLogger(MsgLogger *l) { fLogger = l; }

   private:
      MsgLogger * fLogger;
      MsgLogger& ULog()  { if (fLogger) return *fLogger; return *(fLogger = new MsgLogger("CFMLP_Utils")); } // avoiding control reaches end of non-void function warning

   public:

      ClassDef(MethodCFMlpANN_Utils,0);  // Implementation of Clermond-Ferrand artificial neural network
   };

} // namespace TMVA

#endif
