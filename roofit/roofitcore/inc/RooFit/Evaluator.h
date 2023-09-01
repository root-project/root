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

#ifndef RooFit_Evaluator_h
#define RooFit_Evaluator_h

#include <RooFit/Detail/DataMap.h>
#include <RooHelpers.h>

#include <RConfig.h>

#include <memory>
#include <stack>

class RooAbsArg;

namespace RooFit {

namespace Detail {
class BufferManager;
}

struct NodeInfo;

namespace Detail {
class BufferManager;
}

class Evaluator {
public:
   Evaluator(const RooAbsReal &absReal, bool useGPU=false);
   ~Evaluator();

   std::span<const double> run();
   void setInput(std::string const &name, std::span<const double> inputArray, bool isOnDevice);
   RooArgSet getParameters() const;
   void print(std::ostream &os) const;

private:
   void processVariable(NodeInfo &nodeInfo);
   void setClientsDirty(NodeInfo &nodeInfo);
#ifdef R__HAS_CUDA
   std::span<const double> getValHeterogeneous();
   void markGPUNodes();
   void assignToGPU(NodeInfo &info);
#endif
   void computeCPUNode(const RooAbsArg *node, NodeInfo &info);
   void setOperMode(RooAbsArg *arg, RooAbsArg::OperMode opMode);
   void syncDataTokens();
   void updateOutputSizes();

   std::unique_ptr<Detail::BufferManager> _bufferManager;
   RooAbsReal &_topNode;
   const bool _useGPU = false;
   int _nEvaluations = 0;
   bool _needToUpdateOutputSizes = false;
   RooFit::Detail::DataMap _dataMapCPU;
#ifdef R__HAS_CUDA
   RooFit::Detail::DataMap _dataMapCUDA;
#endif
   std::vector<NodeInfo> _nodes;                                    // the ordered computation graph
   std::stack<RooHelpers::ChangeOperModeRAII> _changeOperModeRAIIs; // for resetting state of computation graph
};

} // end namespace RooFit

#endif
