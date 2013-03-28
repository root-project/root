/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::DecisionTree                                                    *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation of a Decision Tree                                         *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *      Doug Schouten   <dschoute@sfu.ca>        - Simon Fraser U., Canada        *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 *                                                                                *
 **********************************************************************************/

#include "TMVA/CostComplexityPruneTool.h"

#include "TMVA/MsgLogger.h"

#include <fstream>
#include <limits>
#include <math.h>

using namespace TMVA;


//_______________________________________________________________________
CostComplexityPruneTool::CostComplexityPruneTool( SeparationBase* qualityIndex ) : 
   IPruneTool(),
   fLogger(new MsgLogger("CostComplexityPruneTool") )
{
   // the constructor for the cost complexity prunig

   fOptimalK = -1;

   // !! changed from Dougs code. Now use the QualityIndex stored already
   // in the nodes when no "new" QualityIndex calculator is given. Like this
   // I can easily implement the Regression. For Regression, the pruning uses the
   // same sepearation index as in the tree building, hence doesn't need to re-calculate
   // (which would need more info than simply "s" and "b")
   
   fQualityIndexTool = qualityIndex;

   //fLogger->SetMinType( kDEBUG );
   fLogger->SetMinType( kWARNING );
}

//_______________________________________________________________________
CostComplexityPruneTool::~CostComplexityPruneTool( ) {
   // the destructor for the cost complexity prunig
   if(fQualityIndexTool != NULL) delete fQualityIndexTool;
}

//_______________________________________________________________________
PruningInfo*
CostComplexityPruneTool::CalculatePruningInfo( DecisionTree* dt,
                                               const IPruneTool::EventSample* validationSample,
                                               Bool_t isAutomatic )
{

   // the routine that basically "steers" the pruning process. Call the calculation of
   // the pruning sequence, the tree quality and alike..
   
   if( isAutomatic ) SetAutomatic();

   if( dt == NULL || (IsAutomatic() && validationSample == NULL) ) {
      // must have a valid decision tree to prune, and if the prune strength
      // is to be chosen automatically, must have a test sample from
      // which to calculate the quality of the pruned tree(s)
      return NULL;
   }

   Double_t Q = -1.0;
   Double_t W = 1.0;

   if(IsAutomatic()) {
      // run the pruning validation sample through the unpruned tree
      dt->ApplyValidationSample(validationSample);
      W = dt->GetSumWeights(validationSample); // get the sum of weights in the pruning validation sample
      // calculate the quality of the tree in the unpruned case
      Q = dt->TestPrunedTreeQuality();

      Log() << kDEBUG << "Node purity limit is: " << dt->GetNodePurityLimit() << Endl;
      Log() << kDEBUG << "Sum of weights in pruning validation sample: " << W << Endl;
      Log() << kDEBUG << "Quality of tree prior to any pruning is " << Q/W << Endl;
   }

   // store the cost complexity metadata for the decision tree at each node
   try {
      InitTreePruningMetaData((DecisionTreeNode*)dt->GetRoot());
   }
   catch(std::string error) {
      Log() << kERROR << "Couldn't initialize the tree meta data because of error ("
              << error << ")" << Endl;
      return NULL;
   }

   Log() << kDEBUG << "Automatic cost complexity pruning is " << (IsAutomatic()?"on":"off") << "." << Endl;

   try {
      Optimize( dt, W );  // run the cost complexity pruning algorithm
   }
   catch(std::string error) {
      Log() << kERROR << "Error optimzing pruning sequence ("
              << error << ")" << Endl;
      return NULL;
   }

   Log() << kDEBUG << "Index of pruning sequence to stop at: " << fOptimalK << Endl;

   PruningInfo* info = new PruningInfo();


   if(fOptimalK < 0) {
      // no pruning necessary, or wasn't able to compute a sequence
      info->PruneStrength = 0;
      info->QualityIndex = Q/W;
      info->PruneSequence.clear();
      Log() << kINFO << "no proper pruning could be calulated. Tree "   
            <<  dt->GetTreeID() << " will not be pruned. Do not worry if this " 
            << " happens for a few trees " << Endl;
      return info;
   }
   info->QualityIndex = fQualityIndexList[fOptimalK]/W;
   Log() << kDEBUG << " prune until k=" << fOptimalK << " with alpha="<<fPruneStrengthList[fOptimalK]<< Endl;
   for( Int_t i = 0; i < fOptimalK; i++ ){
      info->PruneSequence.push_back(fPruneSequence[i]);
   }
   if( IsAutomatic() ){
      info->PruneStrength = fPruneStrengthList[fOptimalK];
   }
   else {
      info->PruneStrength = fPruneStrength;
   }

   return info;
}

//_______________________________________________________________________
void CostComplexityPruneTool::InitTreePruningMetaData( DecisionTreeNode* n ) {
   // initialise "meta data" for the pruning, like the "costcomplexity", the
   // critical alpha, the minimal alpha down the tree, etc...  for each node!!

   if( n == NULL ) return;

   Double_t s = n->GetNSigEvents();
   Double_t b = n->GetNBkgEvents();
   // set R(t) = N_events*Gini(t) or MisclassificationError(t), etc.
   if (fQualityIndexTool) n->SetNodeR( (s+b)*fQualityIndexTool->GetSeparationIndex(s,b));
   else n->SetNodeR( (s+b)*n->GetSeparationIndex() );

   if(n->GetLeft() != NULL && n->GetRight() != NULL) { // n is an interior (non-leaf) node
      n->SetTerminal(kFALSE);
      // traverse the tree
      InitTreePruningMetaData(n->GetLeft());
      InitTreePruningMetaData(n->GetRight());
      // set |~T_t|
      n->SetNTerminal( n->GetLeft()->GetNTerminal() +
                       n->GetRight()->GetNTerminal());
      // set R(T) = sum[n' in ~T]{ R(n') }
      n->SetSubTreeR( (n->GetLeft()->GetSubTreeR() +
                       n->GetRight()->GetSubTreeR()));
      // set alpha_c, the alpha value at which it becomes advantageaus to prune at node n
      n->SetAlpha( ((n->GetNodeR() - n->GetSubTreeR()) /
                    (n->GetNTerminal() - 1)));

      // G(t) = min( alpha_c, G(l(n)), G(r(n)) )
      // the minimum alpha in subtree rooted at this node
      n->SetAlphaMinSubtree( std::min(n->GetAlpha(), std::min(n->GetLeft()->GetAlphaMinSubtree(),
                                                              n->GetRight()->GetAlphaMinSubtree())));
      n->SetCC(n->GetAlpha());

   } else { // n is a terminal node
      n->SetNTerminal( 1 ); n->SetTerminal( );
      if (fQualityIndexTool) n->SetSubTreeR(((s+b)*fQualityIndexTool->GetSeparationIndex(s,b)));
      else n->SetSubTreeR( (s+b)*n->GetSeparationIndex() );
      n->SetAlpha(std::numeric_limits<double>::infinity( ));
      n->SetAlphaMinSubtree(std::numeric_limits<double>::infinity( ));
      n->SetCC(n->GetAlpha());
   }

//    DecisionTreeNode* R = (DecisionTreeNode*)mdt->GetRoot();
//    Double_t x = R->GetAlphaMinSubtree();
//    Log() << "alphaMin(Root) = " << x << Endl;
}


//_______________________________________________________________________
void CostComplexityPruneTool::Optimize( DecisionTree* dt, Double_t weights ) {
   // after the critical alpha values (at which the corresponding nodes would
   // be pruned away) had been established in the "InitMetaData" we need now:
   // automatic pruning:
   //   find the value of "alpha" for which the test sample gives minimal error,
   //   on the tree with all nodes pruned that have alpha_critital < alpha,
   // fixed parameter pruning
   //

   Int_t k = 1;
   Double_t alpha   = -1.0e10;
   Double_t epsilon = std::numeric_limits<double>::epsilon();

   fQualityIndexList.clear();
   fPruneSequence.clear();
   fPruneStrengthList.clear();

   DecisionTreeNode* R = (DecisionTreeNode*)dt->GetRoot();

   Double_t qmin = 0.0;
   if(IsAutomatic()){
      // initialize the tree quality (actually at this stage, it is the quality of the yet unpruned tree
      qmin = dt->TestPrunedTreeQuality()/weights;
   }

   // now prune the tree in steps until it is gone. At each pruning step, the pruning 
   // takes place at the node that is regarded as the "weakest link".
   // for automatic pruning, at each step, we calculate the current quality of the 
   //     tree and in the end we will prune at the minimum of the tree quality   
   // for the fixed parameter pruing, the cut is simply set at a relative position
   //     in the sequence according to the "length" of the sequence of pruned trees.
   //     100: at the end (pruned until the root node would be the next pruning candidate
   //     50: in the middle of the sequence
   //     etc...
   while(R->GetNTerminal() > 1) { // prune upwards to the root node

      // initialize alpha
      alpha = TMath::Max(R->GetAlphaMinSubtree(), alpha);

      if( R->GetAlphaMinSubtree() >= R->GetAlpha() ) {
         Log() << kDEBUG << "\nCaught trying to prune the root node!" << Endl;
         break;
      }


      DecisionTreeNode* t = R;

      // descend to the weakest link
      while(t->GetAlphaMinSubtree() < t->GetAlpha()) {
//          std::cout << t->GetAlphaMinSubtree() << "  " << t->GetAlpha()<< "  "
//                    << t->GetAlphaMinSubtree()- t->GetAlpha()<<  " t==R?" << int(t == R) << std::endl;
         //      while(  (t->GetAlphaMinSubtree() - t->GetAlpha()) < epsilon)  {
         //         if(TMath::Abs(t->GetAlphaMinSubtree() - t->GetLeft()->GetAlphaMinSubtree())/TMath::Abs(t->GetAlphaMinSubtree()) < epsilon) {
         if(TMath::Abs(t->GetAlphaMinSubtree() - t->GetLeft()->GetAlphaMinSubtree()) < epsilon) {
            t = t->GetLeft();
         } else {
            t = t->GetRight();
         }
      }

      if( t == R ) {
         Log() << kDEBUG << "\nCaught trying to prune the root node!" << Endl;
         break;
      }

      DecisionTreeNode* n = t;

//       Log() << kDEBUG  << "alpha[" << k << "]: " << alpha << Endl;
//       Log() << kDEBUG  << "===========================" << Endl
//               << "Pruning branch listed below the node" << Endl;
//       t->Print( Log() );
//       Log() << kDEBUG << "===========================" << Endl;
//       t->PrintRecPrune( Log() );

      dt->PruneNodeInPlace(t); // prune the branch rooted at node t

      while(t != R) { // go back up the (pruned) tree and recalculate R(T), alpha_c
         t = t->GetParent();
         t->SetNTerminal(t->GetLeft()->GetNTerminal() + t->GetRight()->GetNTerminal());
         t->SetSubTreeR(t->GetLeft()->GetSubTreeR() + t->GetRight()->GetSubTreeR());
         t->SetAlpha((t->GetNodeR() - t->GetSubTreeR())/(t->GetNTerminal() - 1));
         t->SetAlphaMinSubtree(std::min(t->GetAlpha(), std::min(t->GetLeft()->GetAlphaMinSubtree(),
                                                                t->GetRight()->GetAlphaMinSubtree())));
         t->SetCC(t->GetAlpha());
      }
      k += 1;
   
      Log() << kDEBUG << "after this pruning step I would have " << R->GetNTerminal() << " remaining terminal nodes " << Endl;

      if(IsAutomatic()) {
         Double_t q = dt->TestPrunedTreeQuality()/weights;
         fQualityIndexList.push_back(q);
      }
      else {
         fQualityIndexList.push_back(1.0);
      }
      fPruneSequence.push_back(n);
      fPruneStrengthList.push_back(alpha);
   }

   if(fPruneSequence.empty()) {
      fOptimalK = -1;
      return;
   }

   if(IsAutomatic()) {
      k = -1;
      for(UInt_t i = 0; i < fQualityIndexList.size(); i++) {
         if(fQualityIndexList[i] < qmin) {
            qmin = fQualityIndexList[i];
            k = i;
         }
      }
      fOptimalK = k;
   }
   else {
      // regularize the prune strength relative to this tree
      fOptimalK = int(fPruneStrength/100.0 * fPruneSequence.size() );
      Log() << kDEBUG << "SequenzeSize="<<fPruneSequence.size()
            << "  fOptimalK " << fOptimalK << Endl;

   }

   Log() << kDEBUG  << "\n************ Summary for Tree " << dt->GetTreeID() << " *******"  << Endl
         << "Number of trees in the sequence: " << fPruneSequence.size() << Endl;

   Log() << kDEBUG  << "Pruning strength parameters: [";
   for(UInt_t i = 0; i < fPruneStrengthList.size()-1; i++)
      Log() << kDEBUG << fPruneStrengthList[i] << ", ";
   Log() << kDEBUG << fPruneStrengthList[fPruneStrengthList.size()-1] << "]" << Endl;

   Log() << kDEBUG  << "Misclassification rates: [";
   for(UInt_t i = 0; i < fQualityIndexList.size()-1; i++)
      Log() << kDEBUG  << fQualityIndexList[i] << ", ";
   Log() << kDEBUG  << fQualityIndexList[fQualityIndexList.size()-1] << "]"  << Endl;

   Log() << kDEBUG  << "Prune index: " << fOptimalK+1 << Endl;

}

