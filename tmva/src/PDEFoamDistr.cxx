
/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Classes: PDEFoamDistr                                                          *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      The TFDSITR class provides an interface between the Binary search tree    *
 *      and the PDEFoam object.  In order to build-up the foam one needs to       *
 *      calculate the density of events at a given point (sampling during         *
 *      Foam build-up).  The function PDEFoamDistr::Density() does this job.  It  *
 *      uses a binary search tree, filled with training events, in order to       *
 *      provide this density.                                                     *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Tancredi Carli   - CERN, Switzerland                                      *
 *      Dominik Dannheim - CERN, Switzerland                                      *
 *      S. Jadach        - Institute of Nuclear Physics, Cracow, Poland           *
 *      Alexander Voigt  - CERN, Switzerland                                      *
 *      Peter Speckmayer - CERN, Switzerland                                      *
 *                                                                                *
 * Copyright (c) 2008:                                                            *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMath
#include "TMath.h"
#endif

#ifndef ROOT_TMVA_PDEFoamDistr
#include "TMVA/PDEFoamDistr.h"
#endif

ClassImp(TMVA::PDEFoamDistr);

//_____________________________________________________________________
TMVA::PDEFoamDistr::PDEFoamDistr() 
   : TObject(),
     fDim(-1),
     fXmin(0),
     fXmax(0),
     fVolFrac(-1.),
     fBst(NULL),
     fDensityCalc(kEVENT_DENSITY), // default: fill event density to BinarySearchTree
     fSignalClass(1),
     fBackgroundClass(0),
     fLogger( new MsgLogger("PDEFoamDistr"))
{}

//_____________________________________________________________________
TMVA::PDEFoamDistr::~PDEFoamDistr() 
{
   if (fBst)  delete fBst;
   if (fXmin) delete [] fXmin;  fXmin=0;
   if (fXmax) delete [] fXmax;  fXmax=0;
   delete fLogger;
}

//_____________________________________________________________________
TMVA::PDEFoamDistr::PDEFoamDistr(const PDEFoamDistr &distr)
   : TObject(),
     fDim             (distr.fDim),
     fXmin            (distr.fXmin),
     fXmax            (distr.fXmax),
     fVolFrac         (distr.fVolFrac),
     fBst             (distr.fBst),
     fDensityCalc     (kEVENT_DENSITY), // default: fill event density to BinarySearchTree
     fSignalClass     (distr.fSignalClass),
     fBackgroundClass (distr.fBackgroundClass)
{
   // Copy constructor
   Log() << kFATAL << "COPY CONSTRUCTOR NOT IMPLEMENTED" << Endl;
}

//_____________________________________________________________________
void TMVA::PDEFoamDistr::Initialize( Int_t ndim )
{
   // Initialisation procedure of internal foam density.
   // Set dimension to 'ndim' and create BinarySearchTree.

   SetDim(ndim); // set dimension of BST

   if (fBst) delete fBst;
   fBst = new TMVA::BinarySearchTree();

   if (!fBst){
      Log() << kFATAL << "<PDEFoamDistr::Initialize> "
            << "ERROR: an not create binary tree !" << Endl;
   }

   fBst->SetPeriode(fDim);
}

//_____________________________________________________________________
void TMVA::PDEFoamDistr::SetDim(Int_t idim)
{
   // set dimension of distribution

   fDim = idim;
   if (fXmin) delete [] fXmin;
   if (fXmax) delete [] fXmax;
   fXmin = new Float_t[fDim];
   fXmax = new Float_t[fDim];
   return;
}

//_____________________________________________________________________
void TMVA::PDEFoamDistr::FillBinarySearchTree( const Event* ev, EFoamType ft, Bool_t NoNegWeights )
{
   // This method creates an TMVA::Event and inserts it into the
   // binary search tree.
   //
   // If 'NoNegWeights' is true, an event with negative weight will
   // not be filled into the foam.  (Default value: false)

   if((NoNegWeights && ev->GetWeight()<=0) || ev->GetOriginalWeight()==0)
      return;

   TMVA::Event *event = new TMVA::Event(*ev);
   event->SetSignalClass( fSignalClass );

   // set event class and normalization
   if (ft==kSeparate || ft==kDiscr){
      event->SetClass(ev->IsSignal() ? fSignalClass : fBackgroundClass);
   } else if (ft==kMultiTarget){
      // since in multi target regression targets are handled like
      // variables, remove targets and add them to the event variabels
      std::vector<Float_t> targets = ev->GetTargets();
      for (UInt_t i = 0; i < targets.size(); i++)
         event->SetVal(i+ev->GetValues().size(), targets.at(i));
      event->GetTargets().clear();
      event->SetClass(fSignalClass);
   }
   fBst->Insert(event);
}

//_____________________________________________________________________
Double_t TMVA::PDEFoamDistr::Density( Double_t *Xarg, Double_t &event_density )
{
   // This function is needed during the foam buildup.
   // It return a certain density depending on the selected classification
   // or regression options:
   //
   // In case of separated foams (classification) or multi target regression:
   //  - returns event density within volume (specified by VolFrac)
   // In case of unified foams: (classification)
   //  - returns discriminator (N_sig)/(N_sig + N_bg) divided by volume
   //    (specified by VolFrac)
   // In case of mono target regression:
   //  - returns average target value within volume divided by volume
   //    (specified by VolFrac)

   if (!fBst)
      Log() << kFATAL << "<PDEFoamDistr::Density()> Binary tree not found!"<< Endl;

   // make the variable Xarg transform, since Foam only knows about x=[0,1]
   // transformation [0, 1] --> [xmin, xmax]
   for (Int_t idim=0; idim<fDim; idim++)
      Xarg[idim] = VarTransformInvers(idim, Xarg[idim]);

   //create volume around point to be found
   std::vector<Double_t> lb(fDim);
   std::vector<Double_t> ub(fDim);

   // probevolume relative to hypercube with edge length 1:
   static const Double_t probevolume_inv = TMath::Power((fVolFrac/2), fDim);

   // set upper and lower bound for search volume
   for (Int_t idim = 0; idim < fDim; idim++) {
      Double_t volsize=(fXmax[idim] - fXmin[idim]) / fVolFrac;
      lb[idim] = Xarg[idim] - volsize;
      ub[idim] = Xarg[idim] + volsize;
   }

   TMVA::Volume volume(&lb, &ub);                        // volume to search in
   std::vector<const TMVA::BinarySearchTreeNode*> nodes; // BST nodes found

   // do range searching
   fBst->SearchVolume(&volume, &nodes);

   // normalized density: (number of counted events) / volume / (total
   // number of events) should be ~1 on average
   const Double_t count = (Double_t) (nodes.size()); // number of events found

   // store density based on total number of events
   event_density = count * probevolume_inv;

   Double_t weighted_count = 0.; // number of events found (sum of weights!)
   for (UInt_t j=0; j<nodes.size(); j++)
      weighted_count += (nodes.at(j))->GetWeight();

   if (FillDiscriminator()){ // calc number of signal events in nodes
      Double_t N_sig = 0;    // number of target events found
      // now sum over all nodes->IsSignal;
      for (Int_t j=0; j<count; j++){
         N_sig += ((nodes.at(j))->IsSignal()?1.:0.) * (nodes.at(j))->GetWeight();
      }
      return (N_sig/(weighted_count+0.1))*probevolume_inv; // return:  (N_sig/N_total) / (cell_volume)
   }
   else if (FillTarget0()){ // calc sum of weighted target values
      Double_t N_tar = 0;   // number of target events found
      // now sum over all nodes->GetTarget(0);
      for (Int_t j=0; j<count; j++) {
         N_tar += ((nodes.at(j))->GetTargets()).at(0) * ((nodes.at(j))->GetWeight());
      }
      return (N_tar/(weighted_count+0.1))*probevolume_inv; // return:  (N_tar/N_total) / (cell_volume)
   }

   return ((weighted_count+0.1)*probevolume_inv); // return:  N_total(weighted) / cell_volume
}
