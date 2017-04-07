/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : CCPruner                                                              *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description: Cost Complexity Pruning                                           *
 *
 * Author: Doug Schouten (dschoute@sfu.ca)
 *
 *                                                                                *
 * Copyright (c) 2007:                                                            *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Texas at Austin, USA                                                *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#include "TMVA/CCPruner.h"
#include "TMVA/SeparationBase.h"
#include "TMVA/GiniIndex.h"
#include "TMVA/MisClassificationError.h"
#include "TMVA/CCTreeWrapper.h"
#include "TMVA/DataSet.h"

#include "Rtypes.h"

#include <iostream>
#include <fstream>
#include <limits>
#include <math.h>

/*! \class TMVA::CCPruner
\ingroup TMVA
A helper class to prune a decision tree using the Cost Complexity method
(see Classification and Regression Trees by Leo Breiman et al)

### Some definitions:

  - \f$ T_{max} \f$ - the initial, usually highly overtrained tree, that is to be pruned back
  - \f$ R(T) \f$ - quality index (Gini, misclassification rate, or other) of a tree \f$ T \f$
  - \f$ \sim T \f$ - set of terminal nodes in \f$ T \f$
  - \f$ T' \f$ - the pruned subtree of \f$ T_max \f$ that has the best quality index \f$ R(T') \f$
  - \f$ \alpha \f$ - the prune strength parameter in Cost Complexity pruning \f$ (R_{\alpha}(T) = R(T) + \alpha*|\sim T|) \f$

There are two running modes in CCPruner: (i) one may select a prune strength and prune back
the tree \f$ T_{max}\f$  until the criterion:
\f[
 \alpha <  \frac{R(T) - R(t)}{|\sim T_t| - 1}
\f]

is true for all nodes t in \f$ T \f$, or (ii) the algorithm finds the sequence of critical points
\f$ \alpha_k < \alpha_{k+1} ... < \alpha_K \f$ such that \f$ T_K = root(T_{max}) \f$ and then selects the optimally-pruned
subtree, defined to be the subtree with the best quality index for the validation sample.
*/

namespace TMVA {
   class DecisionTree;
}

using namespace TMVA;

////////////////////////////////////////////////////////////////////////////////
/// constructor

CCPruner::CCPruner( DecisionTree* t_max, const EventList* validationSample,
                    SeparationBase* qualityIndex ) :
   fAlpha(-1.0),
   fValidationSample(validationSample),
   fValidationDataSet(NULL),
   fOptimalK(-1)
{
   fTree = t_max;

   if(qualityIndex == NULL) {
      fOwnQIndex = true;
      fQualityIndex = new MisClassificationError();
   }
   else {
      fOwnQIndex = false;
      fQualityIndex = qualityIndex;
   }
   fDebug = kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// constructor

CCPruner::CCPruner( DecisionTree* t_max, const DataSet* validationSample,
                    SeparationBase* qualityIndex ) :
   fAlpha(-1.0),
   fValidationSample(NULL),
   fValidationDataSet(validationSample),
   fOptimalK(-1)
{
   fTree = t_max;

   if(qualityIndex == NULL) {
      fOwnQIndex = true;
      fQualityIndex = new MisClassificationError();
   }
   else {
      fOwnQIndex = false;
      fQualityIndex = qualityIndex;
   }
   fDebug = kTRUE;
}


////////////////////////////////////////////////////////////////////////////////

CCPruner::~CCPruner( )
{
   if(fOwnQIndex) delete fQualityIndex;
   // destructor
}

////////////////////////////////////////////////////////////////////////////////
/// determine the pruning sequence

void CCPruner::Optimize( )
{
   Bool_t HaveStopCondition = fAlpha > 0; // keep pruning the tree until reach the limit fAlpha

   // build a wrapper tree to perform work on
   CCTreeWrapper* dTWrapper = new CCTreeWrapper(fTree, fQualityIndex);

   Int_t    k = 0;
   Double_t epsilon = std::numeric_limits<double>::epsilon();
   Double_t alpha = -1.0e10;

   std::ofstream outfile;
   if (fDebug) outfile.open("costcomplexity.log");
   if(!HaveStopCondition && (fValidationSample == NULL && fValidationDataSet == NULL) ) {
      if (fDebug) outfile << "ERROR: no validation sample, so cannot optimize pruning!" << std::endl;
      delete dTWrapper;
      if (fDebug) outfile.close();
      return;
   }

   CCTreeWrapper::CCTreeNode* R = dTWrapper->GetRoot();
   while(R->GetNLeafDaughters() > 1) { // prune upwards to the root node
      if(R->GetMinAlphaC() > alpha)
         alpha = R->GetMinAlphaC(); // initialize alpha

      if(HaveStopCondition && alpha > fAlpha) break;

      CCTreeWrapper::CCTreeNode* t = R;

      while(t->GetMinAlphaC() < t->GetAlphaC()) { // descend to the weakest link

         if(fabs(t->GetMinAlphaC() - t->GetLeftDaughter()->GetMinAlphaC())/fabs(t->GetMinAlphaC()) < epsilon)
            t = t->GetLeftDaughter();
         else
            t = t->GetRightDaughter();
      }

      if( t == R ) {
         if (fDebug) outfile << std::endl << "Caught trying to prune the root node!" << std::endl;
         break;
      }

      CCTreeWrapper::CCTreeNode* n = t;

      if (fDebug){
         outfile << "===========================" << std::endl
                 << "Pruning branch listed below" << std::endl
                 << "===========================" << std::endl;
         t->PrintRec( outfile );

      }
      if (!(t->GetLeftDaughter()) && !(t->GetRightDaughter()) ) {
         break;
      }
      dTWrapper->PruneNode(t); // prune the branch rooted at node t

      while(t != R) { // go back up the (pruned) tree and recalculate R(T), alpha_c
         t = t->GetMother();
         t->SetNLeafDaughters(t->GetLeftDaughter()->GetNLeafDaughters() + t->GetRightDaughter()->GetNLeafDaughters());
         t->SetResubstitutionEstimate(t->GetLeftDaughter()->GetResubstitutionEstimate() +
                                      t->GetRightDaughter()->GetResubstitutionEstimate());
         t->SetAlphaC((t->GetNodeResubstitutionEstimate() - t->GetResubstitutionEstimate())/(t->GetNLeafDaughters() - 1));
         t->SetMinAlphaC(std::min(t->GetAlphaC(), std::min(t->GetLeftDaughter()->GetMinAlphaC(),
                                                           t->GetRightDaughter()->GetMinAlphaC())));
      }
      k += 1;
      if(!HaveStopCondition) {
         Double_t q;
         if (fValidationDataSet != NULL) q = dTWrapper->TestTreeQuality(fValidationDataSet);
         else q = dTWrapper->TestTreeQuality(fValidationSample);
         fQualityIndexList.push_back(q);
      }
      else {
         fQualityIndexList.push_back(1.0);
      }
      fPruneSequence.push_back(n->GetDTNode());
      fPruneStrengthList.push_back(alpha);
   }

   Double_t qmax = -1.0e6;
   if(!HaveStopCondition) {
      for(UInt_t i = 0; i < fQualityIndexList.size(); i++) {
         if(fQualityIndexList[i] > qmax) {
            qmax = fQualityIndexList[i];
            k = i;
         }
      }
      fOptimalK = k;
   }
   else {
      fOptimalK = fPruneSequence.size() - 1;
   }

   if (fDebug){
      outfile << std::endl << "************ Summary **************"  << std::endl
              << "Number of trees in the sequence: " << fPruneSequence.size() << std::endl;

      outfile << "Pruning strength parameters: [";
      for(UInt_t i = 0; i < fPruneStrengthList.size()-1; i++)
         outfile << fPruneStrengthList[i] << ", ";
      outfile << fPruneStrengthList[fPruneStrengthList.size()-1] << "]" << std::endl;

      outfile << "Misclassification rates: [";
      for(UInt_t i = 0; i < fQualityIndexList.size()-1; i++)
         outfile << fQualityIndexList[i] << ", ";
      outfile << fQualityIndexList[fQualityIndexList.size()-1] << "]"  << std::endl;

      outfile << "Optimal index: " << fOptimalK+1 << std::endl;
      outfile.close();
   }
   delete dTWrapper;
}

////////////////////////////////////////////////////////////////////////////////
/// return the prune strength (=alpha) corresponding to the prune sequence

std::vector<DecisionTreeNode*> CCPruner::GetOptimalPruneSequence( ) const
{
   std::vector<DecisionTreeNode*> optimalSequence;
   if( fOptimalK >= 0 ) {
      for( Int_t i = 0; i < fOptimalK; i++ ) {
         optimalSequence.push_back(fPruneSequence[i]);
      }
   }
   return optimalSequence;
}


