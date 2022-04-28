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

#include <RooAbsReal.h>
#include <RooBatchCompute.h>
#include <RooGlobalFunc.h>
#include <RunContext.h>

#include <RooFit/Detail/Buffers.h>

#include <chrono>
#include <memory>
#include <queue>
#include <stack>
#include <unordered_map>

class RooAbsArg;
class RooAbsCategory;
class RooAbsData;

// Struct to temporarily change the operation mode of a RooAbsArg until it goes
// out of scope.
class ChangeOperModeRAII {
public:
   ChangeOperModeRAII(RooAbsArg *arg, RooAbsArg::OperMode opMode) : _arg{arg}, _oldOpMode(arg->operMode())
   {
      arg->setOperMode(opMode, /*recurse=*/false);
   }
   ~ChangeOperModeRAII() { _arg->setOperMode(_oldOpMode, /*recurse=*/false); }

private:
   RooAbsArg *_arg = nullptr;
   RooAbsArg::OperMode _oldOpMode;
};

namespace ROOT {
namespace Experimental {

class RooFitDriver {
public:
   RooFitDriver(const RooAbsData &data, const RooAbsReal &topNode, RooArgSet const &normSet,
                RooFit::BatchModeOption batchMode, std::string_view rangeName,
                RooAbsCategory const *indexCat = nullptr);

   RooFitDriver(const RooBatchCompute::RunContext &runContext, const RooAbsReal &topNode, RooArgSet const &normSet);

   ~RooFitDriver();
   std::vector<double> getValues();
   double getVal();
   RooAbsReal const &topNode() const { return _topNode; }
   RooArgSet const &parameters() const { return _parameters; }

   class RooAbsRealWrapper final : public RooAbsReal {
   public:
      RooAbsRealWrapper() {}
      RooAbsRealWrapper(RooFitDriver &driver, bool ownsDriver)
         : RooAbsReal{"RooFitDriverWrapper", "RooFitDriverWrapper"}, _driver{&driver}, _ownsDriver{ownsDriver}
      {
      }

      RooAbsRealWrapper(const RooAbsRealWrapper &other, const char *name = 0)
         : RooAbsReal{other, name}, _driver{other._driver}
      {
      }

      ~RooAbsRealWrapper()
      {
         if (_ownsDriver)
            delete _driver;
      }

      TObject *clone(const char *newname) const override { return new RooAbsRealWrapper(*this, newname); }

      double defaultErrorLevel() const override { return _driver->topNode().defaultErrorLevel(); }

      bool getParameters(const RooArgSet * /*observables*/, RooArgSet &outputSet,
                         bool /*stripDisconnected=true*/) const override
      {
         outputSet.add(_driver->parameters());
         return false;
      }

      double getValV(const RooArgSet *) const override { return evaluate(); }

      void applyWeightSquared(bool flag) override
      {
         const_cast<RooAbsReal &>(_driver->topNode()).applyWeightSquared(flag);
      }

   protected:
      double evaluate() const override { return _driver ? _driver->getVal() : 0.0; }

   private:
      RooFitDriver *_driver = nullptr;
      bool _ownsDriver;
   };

   std::unique_ptr<RooAbsReal> makeAbsRealWrapper()
   {
      return std::unique_ptr<RooAbsReal>{new RooAbsRealWrapper{*this, false}};
   }

   // Static method to create a RooAbsRealWrapper that owns a given
   // RooFitDriver passed by smart pointer.
   static std::unique_ptr<RooAbsReal> makeAbsRealWrapper(std::unique_ptr<RooFitDriver> driver)
   {
      return std::unique_ptr<RooAbsReal>{new RooAbsRealWrapper{*driver.release(), true}};
   }

private:
   struct NodeInfo {

      NodeInfo() {}

      // No copying because of the owned CUDA pointers and buffers
      NodeInfo(const NodeInfo &) = delete;
      NodeInfo &operator=(const NodeInfo &) = delete;

      /// Check the servers of a node that has been computed and release it's resources
      /// if they are no longer needed.
      void decrementRemainingClients()
      {
         if (--remClients == 0) {
            buffer.reset();
         }
      }

      RooAbsArg *absArg = nullptr;

      std::unique_ptr<Detail::AbsBuffer> buffer;

      cudaEvent_t *event = nullptr;
      cudaEvent_t *eventStart = nullptr;
      cudaStream_t *stream = nullptr;
      std::chrono::microseconds cpuTime{0};
      std::chrono::microseconds cudaTime{std::chrono::microseconds::max()};
      std::chrono::microseconds timeLaunched{-1};
      int nClients = 0;
      int nServers = 0;
      int remClients = 0;
      int remServers = 0;
      bool computeInScalarMode = false;
      bool computeInGPU = false;
      bool copyAfterEvaluation = false;
      bool fromDataset = false;
      std::size_t outputSize = 1;
      ~NodeInfo()
      {
         if (event)
            RooBatchCompute::dispatchCUDA->deleteCudaEvent(event);
         if (eventStart)
            RooBatchCompute::dispatchCUDA->deleteCudaEvent(eventStart);
         if (stream)
            RooBatchCompute::dispatchCUDA->deleteCudaStream(stream);
      }
   };

   void init(std::map<const TNamed *, RooSpan<const double>> const &dataSpans);

   double getValHeterogeneous();
   void updateMyClients(const RooAbsArg *node);
   void updateMyServers(const RooAbsArg *node);
   void handleIntegral(const RooAbsArg *node);
   std::chrono::microseconds simulateFit(std::chrono::microseconds h2dTime, std::chrono::microseconds d2hTime,
                                         std::chrono::microseconds diffThreshold);
   void markGPUNodes();
   void assignToGPU(const RooAbsArg *node);
   void computeCPUNode(const RooAbsArg *node, NodeInfo &info);

   /// Temporarily change the operation mode of a RooAbsArg until the
   /// RooFitDriver gets deleted.
   void setOperMode(RooAbsArg *arg, RooAbsArg::OperMode opMode)
   {
      if (opMode != arg->operMode()) {
         _changeOperModeRAIIs.emplace(arg, opMode);
      }
   }

   void determineOutputSizes(RooArgSet const &serverSet);

   RooArgSet _parameters;

   const RooFit::BatchModeOption _batchMode = RooFit::BatchModeOption::Off;
   int _getValInvocations = 0;
   double *_cudaMemDataset = nullptr;

   // used for preserving static info about the computation graph
   RooBatchCompute::DataMap _dataMapCPU;
   RooBatchCompute::DataMap _dataMapCUDA;
   const RooAbsReal &_topNode;
   std::unique_ptr<RooArgSet> _normSet;
   std::map<RooAbsArg const *, NodeInfo> _nodeInfos;
   std::map<RooAbsArg const *, NodeInfo> _integralInfos;

   // the ordered computation graph
   std::vector<RooAbsArg *> _orderedNodes;

   // used for preserving resources
   std::vector<double> _nonDerivedValues;
   std::stack<std::vector<double>> _vectorBuffers;

   std::stack<ChangeOperModeRAII> _changeOperModeRAIIs;
}; // end class RooFitDriver

} // end namespace Experimental
} // end namespace ROOT

#endif // ROO_FIT_DRIVER_H
