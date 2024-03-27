/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN 2021
 *   Emmanouil Michalainas, CERN 2021
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFit_Evaluator_h
#define RooFit_Evaluator_h

#include <RooAbsReal.h>
#include <RooFit/EvalContext.h>

#include <RConfig.h>

#include <memory>
#include <stack>

class ChangeOperModeRAII;
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
   Evaluator(const RooAbsReal &absReal, bool useGPU = false);
   ~Evaluator();

   std::span<const double> run();
   void setInput(std::string const &name, std::span<const double> inputArray, bool isOnDevice);
   RooArgSet getParameters() const;
   void print(std::ostream &os);

   void setOffsetMode(RooFit::EvalContext::OffsetMode);

private:
   void processVariable(NodeInfo &nodeInfo);
   void setClientsDirty(NodeInfo &nodeInfo);
   std::span<const double> getValHeterogeneous();
   void markGPUNodes();
   void assignToGPU(NodeInfo &info);
   void computeCPUNode(const RooAbsArg *node, NodeInfo &info);
   void setOperMode(RooAbsArg *arg, RooAbsArg::OperMode opMode);
   void syncDataTokens();
   void updateOutputSizes();

   std::unique_ptr<Detail::BufferManager> _bufferManager;
   RooAbsReal &_topNode;
   const bool _useGPU = false;
   int _nEvaluations = 0;
   bool _needToUpdateOutputSizes = false;
   RooFit::EvalContext _evalContextCPU;
   RooFit::EvalContext _evalContextCUDA;
   std::vector<NodeInfo> _nodes; // the ordered computation graph
   std::stack<std::unique_ptr<ChangeOperModeRAII>> _changeOperModeRAIIs;
};

} // end namespace RooFit

#endif
