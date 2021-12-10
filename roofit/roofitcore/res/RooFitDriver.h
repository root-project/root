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

#include "RooBatchCompute.h"
#include "RooAbsReal.h"

#include "RooFit/Detail/Buffers.h"

#include <chrono>
#include <memory>
#include <queue>
#include <unordered_map>

class RooAbsData;
class RooAbsArg;

namespace ROOT {
namespace Experimental {

class RooFitDriver {
public:
   class Dataset {
   public:
      Dataset(RooAbsData const &data, RooArgSet const &observables, std::string_view rangeName);

      std::size_t size() const { return _nEvents; }
      bool contains(RooAbsArg const *real) const { return _dataSpans.count(real->namePtr()); }
      RooSpan<const double> const &span(RooAbsArg const *real) const { return _dataSpans.at(real->namePtr()); }

   private:
      std::map<const TNamed *, RooSpan<const double>> _dataSpans;
      size_t _nEvents = 0;
      std::stack<std::vector<double>> _buffers;
   };

   RooFitDriver(const RooAbsData &data, const RooAbsReal &topNode, RooArgSet const &normSet,
                RooBatchCompute::BatchMode batchMode, std::string_view rangeName);
   ~RooFitDriver();
   std::vector<double> getValues();
   double getVal();
   RooAbsReal const &topNode() const { return _topNode; }
   std::string const &name() const { return _name; }
   std::string const &title() const { return _title; }
   RooArgSet const &parameters() const { return _parameters; }
   double errorLevel() const { return _topNode.defaultErrorLevel(); }

   class RooAbsRealWrapper final : public RooAbsReal {
   public:
      RooAbsRealWrapper() {}
      RooAbsRealWrapper(RooFitDriver &driver)
         : RooAbsReal{"RooFitDriverWrapper", "RooFitDriverWrapper"}, _driver{&driver}
      {
      }

      RooAbsRealWrapper(const RooAbsRealWrapper &other, const char *name = 0)
         : RooAbsReal{other, name}, _driver{other._driver}
      {
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

   protected:
      double evaluate() const override { return _driver ? _driver->getVal() : 0.0; }

   private:
      RooFitDriver *_driver = nullptr;
   };

   std::unique_ptr<RooAbsReal> makeAbsRealWrapper()
   {
      return std::unique_ptr<RooAbsReal>{new RooAbsRealWrapper{*this}};
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
   void updateMyClients(const RooAbsArg *node);
   void updateMyServers(const RooAbsArg *node);
   void handleIntegral(const RooAbsArg *node);
   std::pair<std::chrono::microseconds, std::chrono::microseconds> memcpyBenchmark();
   std::chrono::microseconds simulateFit(std::chrono::microseconds h2dTime, std::chrono::microseconds d2hTime,
                                         std::chrono::microseconds diffThreshold);
   void markGPUNodes();
   void assignToGPU(const RooAbsArg *node);
   void computeCPUNode(const RooAbsArg *node, NodeInfo &info);

   std::string _name;
   std::string _title;
   RooArgSet _parameters;

   const RooBatchCompute::BatchMode _batchMode = RooBatchCompute::BatchMode::Off;
   int _getValInvocations = 0;
   double *_cudaMemDataset = nullptr;

   // used to get access to the data that we fit to
   const Dataset _dataset;

   // used for preserving static info about the computation graph
   RooBatchCompute::DataMap _dataMapCPU;
   RooBatchCompute::DataMap _dataMapCUDA;
   const RooAbsReal &_topNode;
   RooArgSet _normSet;
   std::unordered_map<const RooAbsArg *, NodeInfo> _nodeInfos;

   // used for preserving resources
   std::vector<double> _nonDerivedValues;
}; // end class RooFitDriver

} // end namespace Experimental
} // end namespace ROOT

#endif // ROO_FIT_DRIVER_H
