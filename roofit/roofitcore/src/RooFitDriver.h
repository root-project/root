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

#include "RooFit/Detail/DataMap.h"
#include <RooGlobalFunc.h>
#include <RooHelpers.h>
#include "RooFit/Detail/Buffers.h"

#include <chrono>
#include <memory>
#include <stack>

class RooAbsArg;
class RooAbsReal;
class RooAbsCategory;
class RooAbsData;
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
   void determineOutputSizes();

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

} // end namespace Experimental
} // end namespace ROOT

#endif
