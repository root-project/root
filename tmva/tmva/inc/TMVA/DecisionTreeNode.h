// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss, Eckhard von Toerne

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : DecisionTreeNode                                                      *
 *                                             *
 *                                                                                *
 * Description:                                                                   *
 *      Node for the Decision Tree                                                *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *      Eckhard von Toerne <evt@physik.uni-bonn.de>  - U. of Bonn, Germany        *
 *                                                                                *
 * Copyright (c) 2009:                                                            *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *       U. of Bonn, Germany                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (see tmva/doc/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_DecisionTreeNode
#define ROOT_TMVA_DecisionTreeNode

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// DecisionTreeNode                                                     //
//                                                                      //
// Node for the Decision Tree                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMVA/Node.h"

#include "TMVA/Version.h"

#include <sstream>
#include <vector>
#include <string>

namespace TMVA {

   class DTNodeTrainingInfo
   {
   public:
   DTNodeTrainingInfo():fSampleMin(),
         fSampleMax(),
         fNodeR(0),fSubTreeR(0),fAlpha(0),fG(0),fNTerminal(0),
         fNB(0),fNS(0),fSumTarget(0),fSumTarget2(0),fCC(0),
         fNSigEvents ( 0 ), fNBkgEvents ( 0 ),
         fNEvents ( -1 ),
         fNSigEvents_unweighted ( 0 ),
         fNBkgEvents_unweighted ( 0 ),
         fNEvents_unweighted ( 0 ),
         fNSigEvents_unboosted ( 0 ),
         fNBkgEvents_unboosted ( 0 ),
         fNEvents_unboosted ( 0 ),
         fSeparationIndex (-1 ),
         fSeparationGain ( -1 )
            {
            }
      std::vector< Float_t >  fSampleMin; ///< the minima for each ivar of the sample on the node during training
      std::vector< Float_t >  fSampleMax; ///< the maxima for each ivar of the sample on the node during training
      Double_t fNodeR;                    ///< node resubstitution estimate, R(t)
      Double_t fSubTreeR;                 ///< R(T) = Sum(R(t) : t in ~T)
      Double_t fAlpha;                    ///< critical alpha for this node
      Double_t fG;                        ///< minimum alpha in subtree rooted at this node
      Int_t    fNTerminal;                ///< number of terminal nodes in subtree rooted at this node
      Double_t fNB;                       ///< sum of weights of background events from the pruning sample in this node
      Double_t fNS;                       ///< ditto for the signal events
      Float_t  fSumTarget;                ///< sum of weight*target  used for the calculation of the variance (regression)
      Float_t  fSumTarget2;               ///< sum of weight*target^2 used for the calculation of the variance (regression)
      Double_t fCC;                       ///< debug variable for cost complexity pruning ..

      Float_t  fNSigEvents;               ///< sum of weights of signal event in the node
      Float_t  fNBkgEvents;               ///< sum of weights of backgr event in the node
      Float_t  fNEvents;                  ///< number of events in that entered the node (during training)
      Float_t  fNSigEvents_unweighted;    ///< sum of signal event in the node
      Float_t  fNBkgEvents_unweighted;    ///< sum of backgr event in the node
      Float_t  fNEvents_unweighted;       ///< number of events in that entered the node (during training)
      Float_t  fNSigEvents_unboosted;     ///< sum of signal event in the node
      Float_t  fNBkgEvents_unboosted;     ///< sum of backgr event in the node
      Float_t  fNEvents_unboosted;        ///< number of events in that entered the node (during training)
      Float_t  fSeparationIndex;          ///< measure of "purity" (separation between S and B) AT this node
      Float_t  fSeparationGain;           ///< measure of "purity", separation, or information gained BY this nodes selection

      // copy constructor
   DTNodeTrainingInfo(const DTNodeTrainingInfo& n) :
      fSampleMin(),fSampleMax(),          ///< Samplemin and max are reset in copy constructor
         fNodeR(n.fNodeR), fSubTreeR(n.fSubTreeR),
         fAlpha(n.fAlpha), fG(n.fG),
         fNTerminal(n.fNTerminal),
         fNB(n.fNB), fNS(n.fNS),
         fSumTarget(0),fSumTarget2(0),    ///< SumTarget reset in copy constructor
         fCC(0),
         fNSigEvents ( n.fNSigEvents ), fNBkgEvents ( n.fNBkgEvents ),
         fNEvents ( n.fNEvents ),
         fNSigEvents_unweighted ( n.fNSigEvents_unweighted ),
         fNBkgEvents_unweighted ( n.fNBkgEvents_unweighted ),
         fNEvents_unweighted ( n.fNEvents_unweighted ),
         fSeparationIndex( n.fSeparationIndex ),
         fSeparationGain ( n.fSeparationGain )
            { }
   };

   class Event;
   class MsgLogger;

   class DecisionTreeNode: public Node {

   public:

      // constructor of an essentially "empty" node floating in space
      DecisionTreeNode ();
      // constructor of a daughter node as a daughter of 'p'
      DecisionTreeNode (Node* p, char pos);

      // copy constructor
      DecisionTreeNode (const DecisionTreeNode &n, DecisionTreeNode* parent = nullptr);

      // destructor
      virtual ~DecisionTreeNode();

      Node* CreateNode() const override { return new DecisionTreeNode(); }

      inline void SetNFisherCoeff(Int_t nvars){fFisherCoeff.resize(nvars);}
      inline UInt_t GetNFisherCoeff() const { return fFisherCoeff.size();}
      // set fisher coefficients
      void SetFisherCoeff(Int_t ivar, Double_t coeff);
      /// get fisher coefficients
      Double_t GetFisherCoeff(Int_t ivar) const {return fFisherCoeff.at(ivar);}

      // test event if it descends the tree at this node to the right
      Bool_t GoesRight( const Event & ) const override;

      // test event if it descends the tree at this node to the left
      Bool_t GoesLeft ( const Event & ) const override;

      /// set index of variable used for discrimination at this node
      void SetSelector( Short_t i) { fSelector = i; }
      /// return index of variable used for discrimination at this node
      Short_t GetSelector() const { return fSelector; }

      /// set the cut value applied at this node
      void  SetCutValue ( Float_t c ) { fCutValue  = c; }
      /// return the cut value applied at this node
      Float_t GetCutValue ( void ) const { return fCutValue;  }

      /// set true: if event variable > cutValue ==> signal , false otherwise
      void SetCutType( Bool_t t   ) { fCutType = t; }
      /// return kTRUE: Cuts select signal, kFALSE: Cuts select bkg
      Bool_t GetCutType( void ) const { return fCutType; }

      /// set node type: 1 signal node, -1 bkg leave, 0 intermediate Node
      void  SetNodeType( Int_t t ) { fNodeType = t;}
      /// return node type: 1 signal node, -1 bkg leave, 0 intermediate Node
      Int_t GetNodeType( void ) const { return fNodeType; }

      /// return  S/(S+B) (purity) at this node (from  training)
      Float_t GetPurity( void ) const { return fPurity;}
      // calculate S/(S+B) (purity) at this node (from  training)
      void SetPurity( void );

      /// set the response of the node (for regression)
      void SetResponse( Float_t r ) { fResponse = r;}

      /// return the response of the node (for regression)
      Float_t GetResponse( void ) const { return fResponse;}

      /// set the RMS of the response of the node (for regression)
      void SetRMS( Float_t r ) { fRMS = r;}

      /// return the RMS of the response of the node (for regression)
      Float_t GetRMS( void ) const { return fRMS;}

      /// set the sum of the signal weights in the node, if traininfo defined
      void SetNSigEvents( Float_t s ) { if(fTrainInfo) fTrainInfo->fNSigEvents = s; }

      /// set the sum of the backgr weights in the node, if traininfo defined
      void SetNBkgEvents( Float_t b ) { if(fTrainInfo) fTrainInfo->fNBkgEvents = b; }

      /// set the number of events that entered the node (during training), if traininfo defined
      void SetNEvents( Float_t nev ){ if(fTrainInfo) fTrainInfo->fNEvents =nev ; }

      /// set the sum of the unweighted signal events in the node, if traininfo defined
      void SetNSigEvents_unweighted( Float_t s ) { if(fTrainInfo) fTrainInfo->fNSigEvents_unweighted = s; }

      /// set the sum of the unweighted backgr events in the node, if traininfo defined
      void SetNBkgEvents_unweighted( Float_t b ) { if(fTrainInfo) fTrainInfo->fNBkgEvents_unweighted = b; }

      /// set the number of unweighted events that entered the node (during training), if traininfo defined
      void SetNEvents_unweighted( Float_t nev ){ if(fTrainInfo) fTrainInfo->fNEvents_unweighted =nev ; }

      /// set the sum of the unboosted signal events in the node, if traininfo defined
      void SetNSigEvents_unboosted( Float_t s ) { if(fTrainInfo) fTrainInfo->fNSigEvents_unboosted = s; }

      /// set the sum of the unboosted backgr events in the node, if traininfo defined
      void SetNBkgEvents_unboosted( Float_t b ) { if(fTrainInfo) fTrainInfo->fNBkgEvents_unboosted = b; }

      /// set the number of unboosted events that entered the node (during training), if traininfo defined
      void SetNEvents_unboosted( Float_t nev ){ if(fTrainInfo) fTrainInfo->fNEvents_unboosted =nev ; }

      /// increment the sum of the signal weights in the node, if traininfo defined
      void IncrementNSigEvents( Float_t s ) { if(fTrainInfo) fTrainInfo->fNSigEvents += s; }

      /// increment the sum of the backgr weights in the node, if traininfo defined
      void IncrementNBkgEvents( Float_t b ) { if(fTrainInfo) fTrainInfo->fNBkgEvents += b; }

      // increment the number of events that entered the node (during training), if traininfo defined
      void IncrementNEvents( Float_t nev ){ if(fTrainInfo) fTrainInfo->fNEvents +=nev ; }

      /// increment the sum of the signal weights in the node, if traininfo defined
      void IncrementNSigEvents_unweighted( ) { if(fTrainInfo) fTrainInfo->fNSigEvents_unweighted += 1; }

      /// increment the sum of the backgr weights in the node, if traininfo defined
      void IncrementNBkgEvents_unweighted( ) { if(fTrainInfo) fTrainInfo->fNBkgEvents_unweighted += 1; }

      /// increment the number of events that entered the node (during training), if traininfo defined
      void IncrementNEvents_unweighted( ){ if(fTrainInfo) fTrainInfo->fNEvents_unweighted +=1 ; }

      /// return the sum of the signal weights in the node, or -1 if traininfo undefined
      Float_t GetNSigEvents( void ) const  { return fTrainInfo ? fTrainInfo->fNSigEvents : -1.; }

      /// return the sum of the backgr weights in the node, or -1 if traininfo undefined
      Float_t GetNBkgEvents( void ) const  { return fTrainInfo ? fTrainInfo->fNBkgEvents : -1.; }

      /// return  the number of events that entered the node (during training), or -1 if traininfo undefined
      Float_t GetNEvents( void ) const  { return fTrainInfo ? fTrainInfo->fNEvents : -1.; }

      // return the sum of unweighted signal weights in the node, or -1 if traininfo undefined
      Float_t GetNSigEvents_unweighted( void ) const  { return fTrainInfo ? fTrainInfo->fNSigEvents_unweighted : -1.; }

      /// return the sum of unweighted backgr weights in the node, or -1 if traininfo undefined
      Float_t GetNBkgEvents_unweighted( void ) const  { return fTrainInfo ? fTrainInfo->fNBkgEvents_unweighted : -1.; }

      /// return  the number of unweighted events that entered the node (during training), or -1 if traininfo undefined
      Float_t GetNEvents_unweighted( void ) const  { return fTrainInfo ? fTrainInfo->fNEvents_unweighted : -1.; }

      /// return the sum of unboosted signal weights in the node, or -1 if traininfo undefined
      Float_t GetNSigEvents_unboosted( void ) const  { return fTrainInfo ? fTrainInfo->fNSigEvents_unboosted : -1.; }

      /// return the sum of unboosted backgr weights in the node, or -1 if traininfo undefined
      Float_t GetNBkgEvents_unboosted( void ) const  { return fTrainInfo ? fTrainInfo->fNBkgEvents_unboosted : -1.; }

      /// return  the number of unboosted events that entered the node (during training), or -1 if traininfo undefined
      Float_t GetNEvents_unboosted( void ) const  { return fTrainInfo ? fTrainInfo->fNEvents_unboosted : -1.; }

      /// set the chosen index, measure of "purity" (separation between S and B) AT this node, if traininfo defined
      void SetSeparationIndex( Float_t sep ){ if(fTrainInfo) fTrainInfo->fSeparationIndex =sep ; }

      /// return the separation index AT this node, or 0 if traininfo undefined
      Float_t GetSeparationIndex( void ) const  { return fTrainInfo ? fTrainInfo->fSeparationIndex : -1.; }

      /// set the separation, or information gained BY this node's selection, if traininfo defined
      void SetSeparationGain( Float_t sep ){ if(fTrainInfo) fTrainInfo->fSeparationGain =sep ; }

      /// return the gain in separation obtained by this node's selection, or -1 if traininfo undefined
      Float_t GetSeparationGain( void ) const  { return fTrainInfo ? fTrainInfo->fSeparationGain : -1.; }

      // printout of the node
      void Print( std::ostream& os ) const override;

      // recursively print the node and its daughters (--> print the 'tree')
      void PrintRec( std::ostream&  os ) const override;

      void AddAttributesToNode(void* node) const override;
      void AddContentToNode(std::stringstream& s) const override;

      // recursively clear the nodes content (S/N etc, but not the cut criteria)
      void ClearNodeAndAllDaughters();

      // get pointers to children, mother in the tree

      // return pointer to the left/right daughter or parent node
      inline DecisionTreeNode* GetLeft( )   const override { return static_cast<DecisionTreeNode*>(fLeft); }
      inline DecisionTreeNode* GetRight( )  const override { return static_cast<DecisionTreeNode*>(fRight); }
      inline DecisionTreeNode* GetParent( ) const override { return static_cast<DecisionTreeNode*>(fParent); }

      // set pointer to the left/right daughter and parent node
      inline void SetLeft  (Node* l) override { fLeft   = l;}
      inline void SetRight (Node* r) override { fRight  = r;}
      inline void SetParent(Node* p) override { fParent = p;}

      /// set the node resubstitution estimate, R(t), for Cost Complexity pruning, if traininfo defined
      inline void SetNodeR( Double_t r ) { if(fTrainInfo) fTrainInfo->fNodeR = r;    }
      /// return the node resubstitution estimate, R(t), for Cost Complexity pruning, or -1 if traininfo undefined
      inline Double_t GetNodeR( ) const  { return fTrainInfo ? fTrainInfo->fNodeR : -1.; }

      /// set the resubstitution estimate, R(T_t), of the tree rooted at this node, if traininfo defined
      inline void SetSubTreeR( Double_t r ) { if(fTrainInfo) fTrainInfo->fSubTreeR = r;    }
      /// return the resubstitution estimate, R(T_t), of the tree rooted at this node, or -1 if traininfo undefined
      inline Double_t GetSubTreeR( ) const  { return fTrainInfo ? fTrainInfo->fSubTreeR : -1.; }

      //                             R(t) - R(T_t)
      // the critical point alpha =  -------------
      //                              |~T_t| - 1
      /// set the critical point alpha, if traininfo defined
      inline void SetAlpha( Double_t alpha ) { if(fTrainInfo) fTrainInfo->fAlpha = alpha; }
      /// return the critical point alpha, or -1 if traininfo undefined
      inline Double_t GetAlpha( ) const      { return fTrainInfo ? fTrainInfo->fAlpha : -1.;  }

      /// set the minimum alpha in the tree rooted at this node, if traininfo defined
      inline void SetAlphaMinSubtree( Double_t g ) { if(fTrainInfo) fTrainInfo->fG = g;    }
      /// return the minimum alpha in the tree rooted at this node, or -1 if traininfo undefined
      inline Double_t GetAlphaMinSubtree( ) const  { return fTrainInfo ? fTrainInfo->fG : -1.; }

      /// set number of terminal nodes in the subtree rooted here, if traininfo defined
      inline void SetNTerminal( Int_t n ) { if(fTrainInfo) fTrainInfo->fNTerminal = n;    }
      /// return number of terminal nodes in the subtree rooted here, or -1 if traininfo undefined
      inline Int_t GetNTerminal( ) const  { return fTrainInfo ? fTrainInfo->fNTerminal : -1.; }

      /// set number of background events from the pruning validation sample, if traininfo defined
      inline void SetNBValidation( Double_t b ) { if(fTrainInfo) fTrainInfo->fNB = b; }
      /// set number of signal events from the pruning validation sample, if traininfo defined
      inline void SetNSValidation( Double_t s ) { if(fTrainInfo) fTrainInfo->fNS = s; }
      /// return number of background events from the pruning validation sample, or -1 if traininfo undefined
      inline Double_t GetNBValidation( ) const  { return fTrainInfo ? fTrainInfo->fNB : -1.; }
      /// return number of signal events from the pruning validation sample, or -1 if traininfo undefined
      inline Double_t GetNSValidation( ) const  { return fTrainInfo ? fTrainInfo->fNS : -1.; }

      /// set sum target, if traininfo defined
      inline void SetSumTarget(Float_t t)  {if(fTrainInfo) fTrainInfo->fSumTarget = t; }
      /// set sum target 2, if traininfo defined
      inline void SetSumTarget2(Float_t t2){if(fTrainInfo) fTrainInfo->fSumTarget2 = t2; }

      /// add to sum target, if traininfo defined
      inline void AddToSumTarget(Float_t t)  {if(fTrainInfo) fTrainInfo->fSumTarget += t; }
      /// add to sum target 2, if traininfo defined
      inline void AddToSumTarget2(Float_t t2){if(fTrainInfo) fTrainInfo->fSumTarget2 += t2; }

      /// return sum target, or -9999 if traininfo undefined
      inline Float_t GetSumTarget()  const {return fTrainInfo? fTrainInfo->fSumTarget : -9999;}
      /// return sum target 2, or -9999 if traininfo undefined
      inline Float_t GetSumTarget2() const {return fTrainInfo? fTrainInfo->fSumTarget2: -9999;}


      // reset the pruning validation data
      void ResetValidationData( );

      /// flag indicates whether this node is terminal
      inline Bool_t IsTerminal() const            { return fIsTerminalNode; }
      inline void SetTerminal( Bool_t s = kTRUE ) { fIsTerminalNode = s;    }
      void PrintPrune( std::ostream& os ) const ;
      void PrintRecPrune( std::ostream& os ) const;

      void     SetCC(Double_t cc);
      /// return CC, or -1 if traininfo undefined
      Double_t GetCC() const {return (fTrainInfo? fTrainInfo->fCC : -1.);}

      Float_t GetSampleMin(UInt_t ivar) const;
      Float_t GetSampleMax(UInt_t ivar) const;
      void     SetSampleMin(UInt_t ivar, Float_t xmin);
      void     SetSampleMax(UInt_t ivar, Float_t xmax);

      static void SetIsTraining(bool on);
      static void SetTmvaVersionCode(UInt_t code);

      static bool IsTraining();
      static UInt_t GetTmvaVersionCode();

      Bool_t ReadDataRecord( std::istream& is, UInt_t tmva_Version_Code = TMVA_VERSION_CODE ) override;
      void ReadAttributes(void* node, UInt_t tmva_Version_Code = TMVA_VERSION_CODE ) override;
      void ReadContent(std::stringstream& s) override;

   protected:

      static MsgLogger& Log();

      static bool fgIsTraining;           ///< static variable to flag training phase in which we need fTrainInfo
      static UInt_t fgTmva_Version_Code;  ///< set only when read from weightfile

      std::vector<Double_t> fFisherCoeff; ///< the fisher coeff (offset at the last element)

      Float_t  fCutValue;                 ///< cut value applied on this node to discriminate bkg against sig
      Bool_t   fCutType;                  ///< true: if event variable > cutValue ==> signal , false otherwise
      Short_t  fSelector;                 ///< index of variable used in node selection (decision tree)

      Float_t  fResponse;                 ///< response value in case of regression
      Float_t  fRMS;                      ///< response RMS of the regression node
      Int_t    fNodeType;                 ///< Type of node: -1 == Bkg-leaf, 1 == Signal-leaf, 0 = internal
      Float_t  fPurity;                   ///< the node purity

      Bool_t   fIsTerminalNode;           ///<! flag to set node as terminal (i.e., without deleting its descendants)

      mutable DTNodeTrainingInfo* fTrainInfo;

   private:

      ClassDefOverride(DecisionTreeNode,0); // Node for the Decision Tree
   };
} // namespace TMVA

#endif
