/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN 2021
 *   Emmanouil Michalainas, CERN 2021
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFit_RooFitDriver_h
#define RooFit_RooFitDriver_h

#include <RooAbsData.h>
#include "RooFit/Detail/DataMap.h"
#include <RooGlobalFunc.h>
#include <RooHelpers.h>
#include <RooRealProxy.h>
#include "RooFit/Detail/Buffers.h"

#include <chrono>
#include <memory>
#include <stack>

class RooAbsArg;
class RooAbsCategory;
class RooSimultaneous;

namespace ROOT {
namespace Experimental {

struct NodeInfo;

class RooFitDriver {
public:
   ////////////////////
   // Enums and aliases

   using DataSpansMap = std::map<RooFit::Detail::DataKey, RooSpan<const double>>;

   //////////////////////////
   // Public member functions

   RooFitDriver(const RooAbsReal &absReal, RooFit::BatchModeOption batchMode = RooFit::BatchModeOption::Cpu);

   void setData(RooAbsData const &data, std::string const &rangeName = "", RooSimultaneous const *simPdf = nullptr,
                bool skipZeroWeights = false, bool takeGlobalObservablesFromData = true);
   void setData(DataSpansMap const &dataSpans);

   ~RooFitDriver();
   std::vector<double> getValues();
   double getVal();
   RooAbsReal &topNode() const;

   void print(std::ostream &os) const;

private:
   ///////////////////////////
   // Private member functions

   double getValHeterogeneous();
   void markGPUNodes();
   void assignToGPU(NodeInfo &info);
   void computeCPUNode(const RooAbsArg *node, NodeInfo &info);
   void setOperMode(RooAbsArg *arg, RooAbsArg::OperMode opMode);

   ///////////////////////////
   // Private member variables

   Detail::BufferManager _bufferManager; // The object managing the different buffers for the intermediate results

   RooAbsReal &_topNode;
   const RooFit::BatchModeOption _batchMode = RooFit::BatchModeOption::Off;
   int _getValInvocations = 0;
   double *_cudaMemDataset = nullptr;

   // used for preserving static info about the computation graph
   RooFit::Detail::DataMap _dataMapCPU;
   RooFit::Detail::DataMap _dataMapCUDA;

   // the ordered computation graph
   std::vector<NodeInfo> _nodes;

   // used for preserving resources
   std::stack<std::vector<double>> _vectorBuffers;

   // RAII structures to reset state of computation graph after driver destruction
   std::stack<RooHelpers::ChangeOperModeRAII> _changeOperModeRAIIs;
};

class RooAbsRealWrapper final : public RooAbsReal {
public:
   RooAbsRealWrapper(std::unique_ptr<RooFitDriver> driver, std::string const &rangeName, RooSimultaneous const *simPdf,
                     bool takeGlobalObservablesFromData);

   RooAbsRealWrapper(const RooAbsRealWrapper &other, const char *name = nullptr);

   TObject *clone(const char *newname) const override { return new RooAbsRealWrapper(*this, newname); }

   double defaultErrorLevel() const override { return _driver->topNode().defaultErrorLevel(); }

   bool getParameters(const RooArgSet *observables, RooArgSet &outputSet, bool stripDisconnected) const override;

   bool setData(RooAbsData &data, bool cloneData) override;

   double getValV(const RooArgSet *) const override { return evaluate(); }

   void applyWeightSquared(bool flag) override
   {
      const_cast<RooAbsReal &>(_driver->topNode()).applyWeightSquared(flag);
   }

   void printMultiline(std::ostream &os, Int_t /*contents*/, bool /*verbose*/ = false,
                       TString /*indent*/ = "") const override
   {
      _driver->print(os);
   }

protected:
   double evaluate() const override { return _driver ? _driver->getVal() : 0.0; }

private:
   std::shared_ptr<RooFitDriver> _driver;
   RooRealProxy _topNode;
   RooAbsData *_data = nullptr;
   RooArgSet _parameters;
   std::string _rangeName;
   RooSimultaneous const *_simPdf = nullptr;
   const bool _takeGlobalObservablesFromData;
};

} // end namespace Experimental
} // end namespace ROOT

#endif
