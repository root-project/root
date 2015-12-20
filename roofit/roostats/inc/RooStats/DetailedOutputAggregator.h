// @(#)root/roostats:$Id$
// Author: Sven Kreiss, Kyle Cranmer   Nov 2010
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_DetailedOutputAggregator
#define ROOSTATS_DetailedOutputAggregator

#ifndef ROOT_TString
#include "TString.h"
#endif


/**

This class is designed to aid in the construction of RooDataSets and RooArgSets, particularly those naturally arising in fitting operations. Typically, the usage of this class is as follows:

1.  create DetailedOutputAggregator instance
2.  use AppendArgSet to add value sets to be stored as one row of the dataset
3.  call CommitSet when an entire row's worth of values has been added
4.  repeat steps 2 and 3 until all rows have been added
5.  call GetAsDataSet to extract result RooDataSet

\ingroup Roostats

*/

class RooAbsCollection; 
class RooFitResult;
class RooDataSet;
class RooArgList;
class RooArgSet; 

namespace RooStats {

   class DetailedOutputAggregator {

   public:

      // Translate the given fit result to a RooArgSet in a generic way.
      // Prefix is prepended to all variable names.
      // Note that the returned set is managed by the user and the user must 
      // explicitly delete all the set content (the returned set does not own the content)
      static RooArgSet *GetAsArgSet(RooFitResult *result, TString prefix="", bool withErrorsAndPulls=false);
      
      DetailedOutputAggregator() {
         fResult = NULL;
         fBuiltSet = NULL;
      }

      // For each variable in aset, prepend prefix to its name and add
      // to the internal store. Note this will not appear in the produced
      // dataset unless CommitSet is called.
      void AppendArgSet(const RooAbsCollection *aset, TString prefix="");

      const RooArgList* GetAsArgList() const {
         // Returns this set of detailed output.
         // Note that the ownership of the returned list is not transfered
         // It is managed by the DetailedOutputAggregator class 
         return fBuiltSet;
      }
      
      // Commit to the result RooDataSet.
      void CommitSet(double weight=1.0);

      RooDataSet *GetAsDataSet(TString name, TString title);

      virtual ~DetailedOutputAggregator();

   private:

      RooDataSet *fResult;
      RooArgList *fBuiltSet;
      
   protected:
      ClassDef(DetailedOutputAggregator,1)
   };
}

#endif
