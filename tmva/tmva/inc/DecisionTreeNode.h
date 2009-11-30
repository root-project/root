// @(#)root/tmva $Id$    
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : DecisionTreeNode                                                      *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Node for the Decision Tree                                                *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         * 
 *      U. of Victoria, Canada                                                    * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
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

#ifndef ROOT_TMVA_Node
#include "TMVA/Node.h"
#endif

#ifndef ROOT_TMVA_Version
#include "TMVA/Version.h"
#endif

#include <vector>
#include <map>
namespace TMVA {
  
   class Event;
   class MsgLogger;

   class DecisionTreeNode: public Node {
    
   public:
    
      // constructor of an essentially "empty" node floating in space
      DecisionTreeNode ();
      // constructor of a daughter node as a daughter of 'p'
      DecisionTreeNode (Node* p, char pos); 
    
      // copy constructor 
      DecisionTreeNode (const DecisionTreeNode &n, DecisionTreeNode* parent = NULL); 
    
      virtual ~DecisionTreeNode(){}

      virtual Node* CreateNode() const { return new DecisionTreeNode(); }
    
      // test event if it decends the tree at this node to the right  
      virtual Bool_t GoesRight( const Event & ) const;
    
      // test event if it decends the tree at this node to the left 
      virtual Bool_t GoesLeft ( const Event & ) const;
    
      // set index of variable used for discrimination at this node
      void SetSelector( Short_t i) { fSelector = i; }
      // return index of variable used for discrimination at this node 
      Short_t GetSelector() const { return fSelector; }
    
    
      // set the cut value applied at this node 
      void  SetCutValue ( Float_t c ) { fCutValue  = c; }
      // return the cut value applied at this node
      Float_t GetCutValue ( void ) const { return fCutValue;  }
    
      // set true: if event variable > cutValue ==> signal , false otherwise
      void SetCutType( Bool_t t   ) { fCutType = t; }
      // return kTRUE: Cuts select signal, kFALSE: Cuts select bkg
      Bool_t GetCutType( void ) const { return fCutType; }
    
      // set node type: 1 signal node, -1 bkg leave, 0 intermediate Node
      void  SetNodeType( Int_t t ) { fNodeType = t;} 
      // return node type: 1 signal node, -1 bkg leave, 0 intermediate Node 
      Int_t GetNodeType( void ) const { return fNodeType; }
    
      //return  S/(S+B) (purity) at this node (from  training)
      Float_t GetPurity( void ) const ;

      //set the response of the node (for regression)
      void SetResponse( Float_t r ) { fResponse = r;}

      //return the response of the node (for regression)
      Float_t GetResponse( void ) const { return fResponse;}

      //set the RMS of the response of the node (for regression)
      void SetRMS( Float_t r ) { fRMS = r;}

      //return the RMS of the response of the node (for regression)
      Float_t GetRMS( void ) const { return fRMS;}

      // set the sum of the signal weights in the node
      void SetNSigEvents( Float_t s ) { fNSigEvents = s; }
    
      // set the sum of the backgr weights in the node
      void SetNBkgEvents( Float_t b ) { fNBkgEvents = b; }
    
      // set the number of events that entered the node (during training)
      void SetNEvents( Float_t nev ){ fNEvents =nev ; }
    
      // set the sum of the unweighted signal events in the node
      void SetNSigEvents_unweighted( Float_t s ) { fNSigEvents_unweighted = s; }
    
      // set the sum of the unweighted backgr events in the node
      void SetNBkgEvents_unweighted( Float_t b ) { fNBkgEvents_unweighted = b; }
    
      // set the number of unweighted events that entered the node (during training)
      void SetNEvents_unweighted( Float_t nev ){ fNEvents_unweighted =nev ; }
    
      // increment the sum of the signal weights in the node
      void IncrementNSigEvents( Float_t s ) { fNSigEvents += s; }
    
      // increment the sum of the backgr weights in the node
      void IncrementNBkgEvents( Float_t b ) { fNBkgEvents += b; }
    
      // increment the number of events that entered the node (during training)
      void IncrementNEvents( Float_t nev ){ fNEvents +=nev ; }
    
      // increment the sum of the signal weights in the node
      void IncrementNSigEvents_unweighted( ) { fNSigEvents_unweighted += 1; }
    
      // increment the sum of the backgr weights in the node
      void IncrementNBkgEvents_unweighted( ) { fNBkgEvents_unweighted += 1; }
    
      // increment the number of events that entered the node (during training)
      void IncrementNEvents_unweighted( ){ fNEvents_unweighted +=1 ; }
    
      // return the sum of the signal weights in the node
      Float_t GetNSigEvents( void ) const  { return fNSigEvents; }
    
      // return the sum of the backgr weights in the node
      Float_t GetNBkgEvents( void ) const  { return fNBkgEvents; }
    
      // return  the number of events that entered the node (during training)
      Float_t GetNEvents( void ) const  { return fNEvents; }
    
      // return the sum of unweighted signal weights in the node
      Float_t GetNSigEvents_unweighted( void ) const  { return fNSigEvents_unweighted; }
    
      // return the sum of unweighted backgr weights in the node
      Float_t GetNBkgEvents_unweighted( void ) const  { return fNBkgEvents_unweighted; }
    
      // return  the number of unweighted events that entered the node (during training)
      Float_t GetNEvents_unweighted( void ) const  { return fNEvents_unweighted; }
    
    
      // set the choosen index, measure of "purity" (separation between S and B) AT this node
      void SetSeparationIndex( Float_t sep ){ fSeparationIndex =sep ; }
      // return the separation index AT this node
      Float_t GetSeparationIndex( void ) const  { return fSeparationIndex; }
    
      // set the separation, or information gained BY this nodes selection
      void SetSeparationGain( Float_t sep ){ fSeparationGain =sep ; }
      // return the gain in separation obtained by this nodes selection
      Float_t GetSeparationGain( void ) const  { return fSeparationGain; }
    
      // printout of the node
      virtual void Print( ostream& os ) const;
    
      // recursively print the node and its daughters (--> print the 'tree')
      virtual void PrintRec( ostream&  os ) const;

      virtual void AddAttributesToNode(void* node) const;
      virtual void AddContentToNode(std::stringstream& s) const;

      // recursively clear the nodes content (S/N etc, but not the cut criteria) 
      void ClearNodeAndAllDaughters();

      // get pointers to children, mother in the tree
      inline DecisionTreeNode* GetLeftDaughter( ) { return dynamic_cast<DecisionTreeNode*>(GetLeft()); }
      inline DecisionTreeNode* GetRightDaughter( ) { return dynamic_cast<DecisionTreeNode*>(GetRight()); }
      inline DecisionTreeNode* GetMother( ) { return dynamic_cast<DecisionTreeNode*>(GetParent()); }
      inline const DecisionTreeNode* GetLeftDaughter( ) const { return dynamic_cast<DecisionTreeNode*>(GetLeft()); }
      inline const DecisionTreeNode* GetRightDaughter( ) const { return dynamic_cast<DecisionTreeNode*>(GetRight()); }
      inline const DecisionTreeNode* GetMother( ) const { return dynamic_cast<DecisionTreeNode*>(GetParent()); }

      ULong_t GetSequence() const {return fSequence;}
    
      void SetSequence(ULong_t s) {fSequence=s;}
    
      // the node resubstitution estimate, R(t), for Cost Complexity pruning
      inline void SetNodeR( Double_t r ) { fNodeR = r;    }
      inline Double_t GetNodeR( ) const  { return fNodeR; }

      // the resubstitution estimate, R(T_t), of the tree rooted at this node
      inline void SetSubTreeR( Double_t r ) { fSubTreeR = r;    }
      inline Double_t GetSubTreeR( ) const  { return fSubTreeR; }

      //                             R(t) - R(T_t)
      // the critical point alpha =  -------------
      //                              |~T_t| - 1
      inline void SetAlpha( Double_t alpha ) { fAlpha = alpha; }
      inline Double_t GetAlpha( ) const      { return fAlpha;  }
    
      // the minimum alpha in the tree rooted at this node
      inline void SetAlphaMinSubtree( Double_t g ) { fG = g;    }
      inline Double_t GetAlphaMinSubtree( ) const  { return fG; }

      // number of terminal nodes in the subtree rooted here
      inline void SetNTerminal( Int_t n ) { fNTerminal = n;    }
      inline Int_t GetNTerminal( ) const  { return fNTerminal; }

      // number of background/signal events from the pruning validation sample
      inline void SetNBValidation( Double_t b ) { fNB = b; }
      inline void SetNSValidation( Double_t s ) { fNS = s; }
      inline Double_t GetNBValidation( ) const  { return fNB; }
      inline Double_t GetNSValidation( ) const  { return fNS; }

    
      inline void SetSumTarget(Float_t t)  {fSumTarget = t; }
      inline void SetSumTarget2(Float_t t2){fSumTarget2 = t2; }

      inline void AddToSumTarget(Float_t t)  {fSumTarget += t; }
      inline void AddToSumTarget2(Float_t t2){fSumTarget2 += t2; }

      inline Float_t GetSumTarget()  const {return fSumTarget; }
      inline Float_t GetSumTarget2() const {return fSumTarget2; }

    
      // reset the pruning validation data
      void ResetValidationData( );

      // flag indicates whether this node is terminal
      inline Bool_t IsTerminal() const            { return fIsTerminalNode; }
      inline void SetTerminal( Bool_t s = kTRUE ) { fIsTerminalNode = s;    }
      void PrintPrune( ostream& os ) const ;
      void PrintRecPrune( ostream& os ) const;

      void     SetCC(Double_t cc) {fCC = cc;};
      Double_t GetCC() const {return fCC;};

      Float_t GetSampleMin(UInt_t ivar) const;
      Float_t GetSampleMax(UInt_t ivar) const;
      void     SetSampleMin(UInt_t ivar, Float_t xmin);
      void     SetSampleMax(UInt_t ivar, Float_t xmax);

   private:

      virtual void ReadAttributes(void* node, UInt_t tmva_Version_Code = TMVA_VERSION_CODE );
      virtual Bool_t ReadDataRecord( istream& is, UInt_t tmva_Version_Code = TMVA_VERSION_CODE );
      virtual void ReadContent(std::stringstream& s);

      Double_t fNodeR;           // node resubstitution estimate, R(t)
      Double_t fSubTreeR;        // R(T) = Sum(R(t) : t in ~T)
      Double_t fAlpha;           // critical alpha for this node
      Double_t fG;               // minimum alpha in subtree rooted at this node
      Int_t    fNTerminal;       // number of terminal nodes in subtree rooted at this node
      Double_t fNB;              // sum of weights of background events from the pruning sample in this node
      Double_t fNS;              // ditto for the signal events

      Float_t  fSumTarget;       // sum of weight*target  used for the calculatio of the variance (regression)
      Float_t  fSumTarget2;      // sum of weight*target^2 used for the calculatio of the variance (regression)
    

      Float_t  fCutValue;        // cut value appplied on this node to discriminate bkg against sig
      Bool_t   fCutType;         // true: if event variable > cutValue ==> signal , false otherwise
      Short_t  fSelector;        // index of variable used in node selection (decision tree) 
    
      Float_t  fNSigEvents;      // sum of weights of signal event in the node
      Float_t  fNBkgEvents;      // sum of weights of backgr event in the node
      Float_t  fNEvents;         // number of events in that entered the node (during training)
    
      Float_t  fNSigEvents_unweighted;      // sum of signal event in the node
      Float_t  fNBkgEvents_unweighted;      // sum of backgr event in the node
      Float_t  fNEvents_unweighted;         // number of events in that entered the node (during training)
    
      Float_t  fSeparationIndex; // measure of "purity" (separation between S and B) AT this node
      Float_t  fSeparationGain;  // measure of "purity", separation, or information gained BY this nodes selection
      Float_t  fResponse;        // response value in case of regression
      Float_t  fRMS;             // response RMS of the regression node 
      Int_t    fNodeType;        // Type of node: -1 == Bkg-leaf, 1 == Signal-leaf, 0 = internal 
    
      ULong_t  fSequence;        // bit coded left right sequence to reach the node

      Bool_t   fIsTerminalNode;    //! flag to set node as terminal (i.e., without deleting its descendants)

      Double_t fCC;              // debug variable for cost complexity pruing .. temporary bla

      std::vector< Float_t >  fSampleMin; // the minima for each ivar of the sample on the node during training
      std::vector< Float_t >  fSampleMax; // the maxima for each ivar of the sample on the node during training



      static MsgLogger* fgLogger;    // static because there is a huge number of nodes...
    
      ClassDef(DecisionTreeNode,0) // Node for the Decision Tree 
    
         };
} // namespace TMVA

#endif 
