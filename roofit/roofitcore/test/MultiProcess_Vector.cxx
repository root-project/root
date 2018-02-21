/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   PB, Patrick Bos,     NL eScience Center, p.bos@esciencecenter.nl        *
 *   IP, Inti Pelupessy,  NL eScience Center, i.pelupessy@esciencecenter.nl  *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#include <cmath>
#include <vector>
#include <map>
#include <exception>

#include <RooRealVar.h>
//#include <Bidi>
//#include <roofit/MultiProcess/Vector.h>

#include "gtest/gtest.h"

class xSquaredPlusBVectorSerial {
 public:
  xSquaredPlusBVectorSerial(double b, std::vector<double> x_init) :
      _b("b", "b", b),
      x(std::move(x_init)),
      result(x.size())
  {}

  virtual void evaluate() {
    // call evaluate_task for each task
    for (std::size_t ix = 0; ix < x.size(); ++ix) {
      result[ix] = std::pow(x[ix], 2) + _b.getVal();
    }
  }

  std::vector<double> get_result() {
    evaluate();
    return result;
  }

 protected:
  RooRealVar _b;
  std::vector<double> x;
  std::vector<double> result;
};


namespace RooFit {
  namespace MultiProcess {


    template <typename Base, typename Derived>
    class Vector : public Base {
      using typename Derived::Message;
     public:
      template <typename... Targs>
      Vector(std::size_t NumCPU, Targs ...args) :
          Base(args...),
          _NumCPU(NumCPU)
      {
        task_indices.reserve(num_tasks_from_cpus());
        for (std::size_t ix = 0; ix < num_tasks_from_cpus(); ++ix) {
          task_indices.emplace_back(ix);
        }


      }

     private:
      virtual void evaluate_task(std::size_t task_index) = 0;
      virtual void sync_worker(std::size_t worker_id) {};


      virtual void process_message(Message m) = 0;

      virtual std::size_t num_tasks_from_cpus() {
        return 1;
      };

//      template <typename M, typename... Targs>
//      void call_method_at_worker(M method, Targs ...args) {
//        method(args...);
//      };

     protected:
      std::vector<std::size_t> task_indices;
      std::size_t _NumCPU;

      template <typename T>
      void enqueue_message(Message m, T a) {};
      template <typename T1, typename T2>
      void enqueue_message(Message m, T1 a1, T2 a2) {};
    };

  }
}


class xSquaredPlusBVectorParallel : public RooFit::MultiProcess::Vector<xSquaredPlusBVectorSerial, xSquaredPlusBVectorParallel> {
  using BASE = RooFit::MultiProcess::Vector<xSquaredPlusBVectorSerial, xSquaredPlusBVectorParallel>;
 public:
  xSquaredPlusBVectorParallel(std::size_t NumCPU, double b_init, std::vector<double> x_init) :
      BASE(NumCPU, b_init, x_init) // NumCPU stands for everything that defines the parallelization behaviour (number of cpu, strategy, affinity etc)
  {}

  enum class Message {set_b_worker, retrieve_task_elements};
//  std::map<Message, void (*)()>

//  void evaluate() override {
//    // sync remote first: local b -> workers
//    sync();
//    // choose parallel strategy from multiprocess vector
//    // and don't pass evaluate_task (because virtual)
//  }

  void set_b_workers(double b) {}


  void sync() {
    // implementation defines sync, in this case update b only
//    for (auto worker_id : workers) {
//      sync_worker(worker_id);
//    }
  }


 private:
  // gets called from the server loop (on worker) when a message arrives
  void process_message(Message m) override {
    // do nothing for now
    switch (m) {
      case Message::set_b_worker: {
//        double b;
//        *_pipe >> b;
//        _b.setVal(b);
//        set_b_workers(_b.getVal());
        break;
      }
      case Message::retrieve_task_elements: {

        break;
      }
      default: {
        throw std::runtime_error("go away");
      }
    }
  }

  void sync_worker(std::size_t worker_id) override {
    enqueue_message(Message::set_b_worker, _b.getVal());
//    set_b_worker(_b, worker_id);
  }

  void evaluate_task(std::size_t task_index) {
    result[task_index] = std::pow(x[task_index], 2) + _b.getVal();
  } // if serial implementation doesn't define evaluate_task -> implement here
};


TEST(MultiProcess_Vector, xSquaredPlusB) {
  // Simple test case: calculate x^2 + b, where x is a vector. This case does
  // both a simple calculation (squaring the input vector x) and represents
  // handling of state updates in b.
  std::vector<double> x {0, 1, 2, 3};
  double b_initial = 3.;

  xSquaredPlusBVectorSerial x_sq_plus_b(b_initial, x);

  auto y = x_sq_plus_b.get_result();
  std::vector<double> y_expected {3, 4, 7, 12};

  EXPECT_EQ(y[0], y_expected[0]);
  EXPECT_EQ(y[1], y_expected[1]);
  EXPECT_EQ(y[2], y_expected[2]);
  EXPECT_EQ(y[3], y_expected[3]);

  std::size_t NumCPU = 1;
  xSquaredPlusBVectorParallel x_sq_plus_b_parallel(NumCPU, b_initial, x);

  auto y_parallel = x_sq_plus_b_parallel.get_result();

  EXPECT_EQ(y_parallel[0], y_expected[0]);
  EXPECT_EQ(y_parallel[1], y_expected[1]);
  EXPECT_EQ(y_parallel[2], y_expected[2]);
  EXPECT_EQ(y_parallel[3], y_expected[3]);
}
