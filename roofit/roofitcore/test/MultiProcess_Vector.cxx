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

#include <MultiProcess/messages.h>
#include <MultiProcess/TaskManager.h>
#include <MultiProcess/Vector.h>

#include <cstdlib>  // std::_Exit
#include <cmath>
#include <vector>
#include <map>
#include <exception>
#include <memory>  // make_shared
#include <numeric> // accumulate
#include <tuple>   // for google test Combine in parameterized test, and std::tie

#include <RooRealVar.h>
#include <MultiProcess/BidirMMapPipe.h>
#include <ROOT/RMakeUnique.hxx>
#include <RooRandom.h>

// for NLL tests
#include <RooWorkspace.h>
#include <RooAbsPdf.h>
#include <RooDataSet.h>
#include <RooNLLVar.h>
#include <RooMinimizer.h>
#include <RooFitResult.h>

#include "gtest/gtest.h"

class xSquaredPlusBVectorSerial {
 public:
  xSquaredPlusBVectorSerial(double b, std::vector<double> x_init) :
      _b("b", "b", b),
      x(std::move(x_init)),
      result(x.size()) {}

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


using RooFit::MultiProcess::JobTask;

class xSquaredPlusBVectorParallel : public RooFit::MultiProcess::Vector<xSquaredPlusBVectorSerial> {
 public:
  xSquaredPlusBVectorParallel(std::size_t NumCPU, double b_init, std::vector<double> x_init) :
      RooFit::MultiProcess::Vector<xSquaredPlusBVectorSerial>(NumCPU, b_init,
           x_init) // NumCPU stands for everything that defines the parallelization behaviour (number of cpu, strategy, affinity etc)
  {}

  void init_vars() override {
    // we don't do anything here, this is just a test class which we don't use for testing parameter updates
  }

  void evaluate() override {
    if (get_manager()->is_master()) {
      // start work mode
      get_manager()->set_work_mode(true);

      // master fills queue with tasks
      retrieved = false;
      for (std::size_t task_id = 0; task_id < x.size(); ++task_id) {
        JobTask job_task(id, task_id);
        get_manager()->to_queue(job_task);
      }

      // wait for task results back from workers to master
      gather_worker_results();

      // end work mode
      get_manager()->set_work_mode(false);

      // put task results in desired container (same as used in serial class)
      for (std::size_t task_id = 0; task_id < x.size(); ++task_id) {
        result[task_id] = results[task_id];
      }
    }
  }


 private:
  void evaluate_task(std::size_t task) override {
    assert(get_manager()->is_worker());
    result[task] = std::pow(x[task], 2) + _b.getVal();
  }

  double get_task_result(std::size_t task) override {
    assert(get_manager()->is_worker());
    return result[task];
  }

};

class MultiProcessVectorSingleJob : public ::testing::TestWithParam<std::size_t> {
  // You can implement all the usual fixture class members here.
  // To access the test parameter, call GetParam() from class
  // TestWithParam<T>.
};


class Hex {
 public:
  explicit Hex(double n) : number_(n) {}
  operator double() const { return number_; }
  bool operator==(const Hex& other) {
    return double(*this) == double(other);
  }

 private:
  double number_;
};

::std::ostream& operator<<(::std::ostream& os, const Hex& hex) {
  return os << std::hexfloat << double(hex) << std::defaultfloat;  // whatever needed to print bar to os
}


TEST_P(MultiProcessVectorSingleJob, getResult) {
  // Simple test case: calculate x^2 + b, where x is a vector. This case does
  // both a simple calculation (squaring the input vector x) and represents
  // handling of state updates in b.
  std::vector<double> x{0, 1, 2, 3};
  double b_initial = 3.;

  // start serial test

  xSquaredPlusBVectorSerial x_sq_plus_b(b_initial, x);

  auto y = x_sq_plus_b.get_result();
  std::vector<double> y_expected{3, 4, 7, 12};

  EXPECT_EQ(Hex(y[0]), Hex(y_expected[0]));
  EXPECT_EQ(Hex(y[1]), Hex(y_expected[1]));
  EXPECT_EQ(Hex(y[2]), Hex(y_expected[2]));
  EXPECT_EQ(Hex(y[3]), Hex(y_expected[3]));

  std::size_t NumCPU = GetParam();

  // start parallel test

  xSquaredPlusBVectorParallel x_sq_plus_b_parallel(NumCPU, b_initial, x);

  auto y_parallel = x_sq_plus_b_parallel.get_result();

  EXPECT_EQ(Hex(y_parallel[0]), Hex(y_expected[0]));
  EXPECT_EQ(Hex(y_parallel[1]), Hex(y_expected[1]));
  EXPECT_EQ(Hex(y_parallel[2]), Hex(y_expected[2]));
  EXPECT_EQ(Hex(y_parallel[3]), Hex(y_expected[3]));
}


INSTANTIATE_TEST_CASE_P(NumberOfWorkerProcesses,
                        MultiProcessVectorSingleJob,
                        ::testing::Values(1,2,3));


class MultiProcessVectorMultiJob : public ::testing::TestWithParam<std::size_t> {};

TEST_P(MultiProcessVectorMultiJob, getResult) {
  // Simple test case: calculate x^2 + b, where x is a vector. This case does
  // both a simple calculation (squaring the input vector x) and represents
  // handling of state updates in b.
  std::vector<double> x{0, 1, 2, 3};
  double b_initial = 3.;

  std::vector<double> y_expected{3, 4, 7, 12};

  std::size_t NumCPU = GetParam();

  // define jobs
  xSquaredPlusBVectorParallel x_sq_plus_b_parallel(NumCPU, b_initial, x);
  xSquaredPlusBVectorParallel x_sq_plus_b_parallel2(NumCPU, b_initial + 1, x);

  // do stuff
  auto y_parallel = x_sq_plus_b_parallel.get_result();
  auto y_parallel2 = x_sq_plus_b_parallel2.get_result();

  EXPECT_EQ(Hex(y_parallel[0]), Hex(y_expected[0]));
  EXPECT_EQ(Hex(y_parallel[1]), Hex(y_expected[1]));
  EXPECT_EQ(Hex(y_parallel[2]), Hex(y_expected[2]));
  EXPECT_EQ(Hex(y_parallel[3]), Hex(y_expected[3]));

  EXPECT_EQ(Hex(y_parallel2[0]), Hex(y_expected[0] + 1));
  EXPECT_EQ(Hex(y_parallel2[1]), Hex(y_expected[1] + 1));
  EXPECT_EQ(Hex(y_parallel2[2]), Hex(y_expected[2] + 1));
  EXPECT_EQ(Hex(y_parallel2[3]), Hex(y_expected[3] + 1));
}


INSTANTIATE_TEST_CASE_P(NumberOfWorkerProcesses,
                        MultiProcessVectorMultiJob,
                        ::testing::Values(2,1,3));


enum class RooNLLVarTask {
  all_events,
  single_event,
  bulk_partition,
  interleave
};

// for debugging:
std::ostream& operator<<(std::ostream& out, const RooNLLVarTask value){
  const char* s = 0;
#define PROCESS_VAL(p) case(p): s = #p; break;
  switch(value){
    PROCESS_VAL(RooNLLVarTask::all_events);
    PROCESS_VAL(RooNLLVarTask::single_event);
    PROCESS_VAL(RooNLLVarTask::bulk_partition);
    PROCESS_VAL(RooNLLVarTask::interleave);
  }
#undef PROCESS_VAL
  return out << s;
}


template <typename C>
typename C::value_type sum_kahan(const C& container) {
  using ValueType = typename C::value_type;
  ValueType sum = 0, carry = 0;
  for (auto element : container) {
    ValueType y = element - carry;
    ValueType t = sum + y;
    carry = (t - sum) - y;
    sum = t;
  }
  return sum;
}

template <typename IndexType, typename ValueType>
ValueType sum_kahan(const std::map<IndexType, ValueType>& map) {
  ValueType sum = 0, carry = 0;
  for (auto element : map) {
    ValueType y = element.second - carry;
    ValueType t = sum + y;
    carry = (t - sum) - y;
    sum = t;
  }
  return sum;
}

template <typename C>
std::pair<typename C::value_type, typename C::value_type> sum_of_kahan_sums(const C& sum_values, const C& sum_carrys) {
  using ValueType = typename C::value_type;
  ValueType sum = 0, carry = 0;
  for (std::size_t ix = 0; ix < sum_values.size(); ++ix) {
    ValueType y = sum_values[ix];
    carry += sum_carrys[ix];
    y -= carry;
    const ValueType t = sum + y;
    carry = (t - sum) - y;
    sum = t;
  }
  return std::pair<ValueType, ValueType>(sum, carry);
}



class MPRooNLLVar : public RooFit::MultiProcess::Vector<RooNLLVar> {
 public:
  // use copy constructor for the RooNLLVar part
  MPRooNLLVar(std::size_t NumCPU, RooNLLVarTask task_mode, const RooNLLVar& nll) :
      RooFit::MultiProcess::Vector<RooNLLVar>(NumCPU, nll),
      mp_task_mode(task_mode)
  {
    if (_gofOpMode == RooAbsTestStatistic::GOFOpMode::MPMaster) {
      throw std::logic_error("Cannot create MPRooNLLVar based on a multi-CPU enabled RooNLLVar! The use of the BidirMMapPipe by MPFE in RooNLLVar conflicts with the use of BidirMMapPipe by MultiProcess classes.");
    }

    _vars = RooListProxy("vars", "vars", this);
    init_vars();
    switch (mp_task_mode) {
      case RooNLLVarTask::all_events: {
        N_tasks = 1;
        break;
      }
      case RooNLLVarTask::single_event: {
        N_tasks = static_cast<std::size_t>(_data->numEntries());
        break;
      }
      case RooNLLVarTask::bulk_partition:
      case RooNLLVarTask::interleave: {
        N_tasks = NumCPU;
        break;
      }
    }

    double_const_methods["getCarry"] = &MPRooNLLVar::getCarry;
  }

  void init_vars() override {
    // Empty current lists
    _vars.removeAll() ;
    _saveVars.removeAll() ;

    // Retrieve non-constant parameters
    auto vars = std::make_unique<RooArgSet>(*getParameters(RooArgSet()));
    RooArgList varList(*vars);

    // Save in lists
    _vars.add(varList);
    _saveVars.addClone(varList);
  }


  void update_parameters() {
    if (get_manager()->is_master()) {
      for (std::size_t ix = 0u; ix < static_cast<std::size_t>(_vars.getSize()); ++ix) {
        bool valChanged = !_vars[ix].isIdentical(_saveVars[ix], kTRUE);
        bool constChanged = (_vars[ix].isConstant() != _saveVars[ix].isConstant());

        if (valChanged || constChanged) {
          if (constChanged) {
            ((RooRealVar *) &_saveVars[ix])->setConstant(_vars[ix].isConstant());
          }
          // TODO: Check with Wouter why he uses copyCache in MPFE; makes it very difficult to extend, because copyCache is protected (so must be friend). Moved setting value to if-block below.
//          _saveVars[ix].copyCache(&_vars[ix]);

          // send message to queue (which will relay to workers)
          RooAbsReal * rar_val = dynamic_cast<RooAbsReal *>(&_vars[ix]);
          if (rar_val) {
            Double_t val = rar_val->getVal();
            dynamic_cast<RooRealVar *>(&_saveVars[ix])->setVal(val);
            RooFit::MultiProcess::M2Q msg = RooFit::MultiProcess::M2Q::update_real;
            Bool_t isC = _vars[ix].isConstant();
            *get_manager()->get_queue_pipe() << msg << id << ix << val << isC << RooFit::BidirMMapPipe::flush;
          }
          // TODO: implement category handling
//            } else if (dynamic_cast<RooAbsCategory*>(var)) {
//              M2Q msg = M2Q::update_cat ;
//              UInt_t cat_ix = ((RooAbsCategory*)var)->getIndex();
//              *_pipe << msg << ix << cat_ix;
//            }
        }
      }
    }
  }

  // the const is inherited from RooAbsTestStatistic::evaluate. We are not
  // actually const though, so we use a horrible hack.
  Double_t evaluate() const override {
    return const_cast<MPRooNLLVar*>(this)->evaluate_non_const();
  }

  Double_t evaluate_non_const() {
    if (get_manager()->is_master()) {
      // update parameters that changed since last calculation (or creation if first time)
      update_parameters();

      // activate work mode
      get_manager()->set_work_mode(true);

      // master fills queue with tasks
      retrieved = false;
      for (std::size_t ix = 0; ix < N_tasks; ++ix) {
        JobTask job_task(id, ix);
        get_manager()->to_queue(job_task);
      }

      // wait for task results back from workers to master
      gather_worker_results();

      // end work mode
      get_manager()->set_work_mode(false);

      // put the results in vectors for calling sum_of_kahan_sums (TODO: make a map-friendly sum_of_kahan_sums)
      std::vector<double> results_vec, carrys_vec;
      for (auto const &item : results) {
        results_vec.emplace_back(item.second);
        carrys_vec.emplace_back(carrys[item.first]);
      }

      // sum task results
      std::tie(result, carry) = sum_of_kahan_sums(results_vec, carrys_vec);
    }
    return result;
  }


  // --- RESULT LOGISTICS ---

  void send_back_task_result_from_worker(std::size_t task) override {
    result = get_task_result(task);
    carry = getCarry();
    get_manager()->send_from_worker_to_queue(id, task, result, carry);
  }

  void receive_task_result_on_queue(std::size_t task, std::size_t worker_id) override {
    result = get_manager()->receive_from_worker_on_queue<double>(worker_id);
    carry = get_manager()->receive_from_worker_on_queue<double>(worker_id);
    results[task] = result;
    carrys[task] = carry;
  }

  void send_back_results_from_queue_to_master() override {
    get_manager()->send_from_queue_to_master(results.size());
    for (auto const &item : results) {
      get_manager()->send_from_queue_to_master(item.first, item.second, carrys[item.first]);
    }
  }

  void clear_results() override {
    // empty results caches
    results.clear();
    carrys.clear();
  }

  void receive_results_on_master() override {
    std::size_t N_job_tasks = get_manager()->receive_from_queue_on_master<std::size_t>();
    for (std::size_t task_ix = 0ul; task_ix < N_job_tasks; ++task_ix) {
      std::size_t task_id = get_manager()->receive_from_queue_on_master<std::size_t>();
      results[task_id] = get_manager()->receive_from_queue_on_master<double>();
      carrys[task_id] = get_manager()->receive_from_queue_on_master<double>();
    }
  }

  // --- END OF RESULT LOGISTICS ---


 private:
  void evaluate_task(std::size_t task) override {
    assert(get_manager()->is_worker());
    std::size_t N_events = static_cast<std::size_t>(_data->numEntries());
    // "default" values (all events in one task)
    std::size_t first = task;
    std::size_t last  = N_events;
    std::size_t step  = 1;
    switch (mp_task_mode) {
      case RooNLLVarTask::all_events: {
        // default values apply
        break;
      }
      case RooNLLVarTask::single_event: {
        last = task + 1;
        break;
      }
      case RooNLLVarTask::bulk_partition: {
        first = N_events * task / N_tasks;
        last  = N_events * (task + 1) / N_tasks;
        break;
      }
      case RooNLLVarTask::interleave: {
        step = N_tasks;
        break;
      }
    }

    result = evaluatePartition(first, last, step);
  }

  double get_task_result(std::size_t /*task*/) override {
    // TODO: this is quite ridiculous, having a get_task_result without task
    // argument. We should have a cache, e.g. a map, that gives the result for
    // a given task. The caller (usually send_back_task_result_from_worker) can
    // then decide whether to erase the value from the cache to keep it clean.
    assert(get_manager()->is_worker());
    return result;
  }

  std::map<std::size_t, double> carrys;
  double result = 0;
  double carry = 0;
  std::size_t N_tasks = 0;
  RooNLLVarTask mp_task_mode;
};





TEST(MPFEnll, getVal) {
  // check whether MPFE produces the same results when using different NumCPU or mode.
  // this defines the baseline against which we compare our MP NLL
  RooRandom::randomGenerator()->SetSeed(3);
  // N.B.: it passes on seeds 1 and 2

  RooWorkspace w;
  w.factory("Gaussian::g(x[-5,5],mu[0,-3,3],sigma[1])");
  auto x = w.var("x");
  RooAbsPdf *pdf = w.pdf("g");
  RooRealVar *mu = w.var("mu");
  RooDataSet *data = pdf->generate(RooArgSet(*x), 10000);
  double results[4];

  RooArgSet values = RooArgSet(*mu, *pdf);

  auto nll1 = pdf->createNLL(*data, RooFit::NumCPU(1));
  results[0] = nll1->getVal();
  delete nll1;
  auto nll2 = pdf->createNLL(*data, RooFit::NumCPU(2));
  results[1] = nll2->getVal();
  delete nll2;
  auto nll3 = pdf->createNLL(*data, RooFit::NumCPU(3));
  results[2] = nll3->getVal();
  delete nll3;
  auto nll4 = pdf->createNLL(*data, RooFit::NumCPU(4));
  results[3] = nll4->getVal();
  delete nll4;
  auto nll1b = pdf->createNLL(*data, RooFit::NumCPU(1));
  auto result1b = nll1b->getVal();
  delete nll1b;
  auto nll2b = pdf->createNLL(*data, RooFit::NumCPU(2));
  auto result2b = nll2b->getVal();
  delete nll2b;

  auto nll1_mpfe = pdf->createNLL(*data, RooFit::NumCPU(-1));
  auto result1_mpfe = nll1_mpfe->getVal();
  delete nll1_mpfe;

  auto nll1_interleave = pdf->createNLL(*data, RooFit::NumCPU(1, 1));
  auto result_interleave1 = nll1_interleave->getVal();
  delete nll1_interleave;
  auto nll2_interleave = pdf->createNLL(*data, RooFit::NumCPU(2, 1));
  auto result_interleave2 = nll2_interleave->getVal();
  delete nll2_interleave;
  auto nll3_interleave = pdf->createNLL(*data, RooFit::NumCPU(3, 1));
  auto result_interleave3 = nll3_interleave->getVal();
  delete nll3_interleave;

  EXPECT_DOUBLE_EQ(Hex(results[0]), Hex(results[1]));
  EXPECT_DOUBLE_EQ(Hex(results[0]), Hex(results[2]));
  EXPECT_DOUBLE_EQ(Hex(results[0]), Hex(results[3]));

  EXPECT_DOUBLE_EQ(Hex(results[0]), Hex(result1b));
  EXPECT_DOUBLE_EQ(Hex(results[1]), Hex(result2b));
  EXPECT_DOUBLE_EQ(Hex(results[0]), Hex(result1_mpfe));

  EXPECT_DOUBLE_EQ(Hex(results[0]), Hex(result_interleave1));
  EXPECT_DOUBLE_EQ(Hex(results[0]), Hex(result_interleave2));
  EXPECT_DOUBLE_EQ(Hex(results[0]), Hex(result_interleave3));
}


class MultiProcessVectorNLL : public ::testing::TestWithParam<std::tuple<std::size_t, RooNLLVarTask, std::size_t>> {};


TEST_P(MultiProcessVectorNLL, getVal) {
  // Real-life test: calculate a NLL using event-based parallelization. This
  // should replicate RooRealMPFE results.
  RooRandom::randomGenerator()->SetSeed(std::get<2>(GetParam()));
  RooWorkspace w;
  w.factory("Gaussian::g(x[-5,5],mu[0,-3,3],sigma[1])");
  auto x = w.var("x");
  RooAbsPdf *pdf = w.pdf("g");
  RooDataSet *data = pdf->generate(RooArgSet(*x), 10000);
  auto nll = pdf->createNLL(*data);

  auto nominal_result = nll->getVal();

  std::size_t NumCPU = std::get<0>(GetParam());
  RooNLLVarTask mp_task_mode = std::get<1>(GetParam());

  MPRooNLLVar nll_mp(NumCPU, mp_task_mode, *dynamic_cast<RooNLLVar*>(nll));

  auto mp_result = nll_mp.getVal();

  EXPECT_DOUBLE_EQ(Hex(nominal_result), Hex(mp_result));
  if (HasFailure()) {
    std::cout << "failed test had parameters NumCPU = " << NumCPU << ", task_mode = " << mp_task_mode << ", seed = " << std::get<2>(GetParam()) << std::endl;
  }
}


TEST_P(MultiProcessVectorNLL, setVal) {
  // calculate the NLL twice with different parameters

  RooRandom::randomGenerator()->SetSeed(std::get<2>(GetParam()));
  RooWorkspace w;
  w.factory("Gaussian::g(x[-5,5],mu[0,-3,3],sigma[1])");
  auto x = w.var("x");
  RooAbsPdf *pdf = w.pdf("g");
  RooDataSet *data = pdf->generate(RooArgSet(*x), 10000);
  auto nll = pdf->createNLL(*data);

  std::size_t NumCPU = std::get<0>(GetParam());
  RooNLLVarTask mp_task_mode = std::get<1>(GetParam());

  MPRooNLLVar nll_mp(NumCPU, mp_task_mode, *dynamic_cast<RooNLLVar*>(nll));

  // calculate first results
  nll->getVal();
  nll_mp.getVal();

  w.var("mu")->setVal(2);

  // calculate second results after parameter change
  auto nominal_result2 = nll->getVal();
  auto mp_result2 = nll_mp.getVal();

  EXPECT_DOUBLE_EQ(Hex(nominal_result2), Hex(mp_result2));
  if (HasFailure()) {
    std::cout << "failed test had parameters NumCPU = " << NumCPU << ", task_mode = " << mp_task_mode << ", seed = " << std::get<2>(GetParam()) << std::endl;
  }
}


INSTANTIATE_TEST_CASE_P(NworkersModeSeed,
                        MultiProcessVectorNLL,
                        ::testing::Combine(::testing::Values(1,2,3),  // number of workers
                                           ::testing::Values(RooNLLVarTask::all_events,
                                                             RooNLLVarTask::single_event,
                                                             RooNLLVarTask::bulk_partition,
                                                             RooNLLVarTask::interleave),
                                           ::testing::Values(2,3)));  // random seed




class NLLMultiProcessVsMPFE : public ::testing::TestWithParam<std::tuple<std::size_t, RooNLLVarTask, std::size_t>> {};

TEST_P(NLLMultiProcessVsMPFE, getVal) {
  // Compare our MP NLL to actual RooRealMPFE results using the same strategies.

  // parameters
  std::size_t NumCPU = std::get<0>(GetParam());
  RooNLLVarTask mp_task_mode = std::get<1>(GetParam());
  std::size_t seed = std::get<2>(GetParam());

  RooRandom::randomGenerator()->SetSeed(seed);

  RooWorkspace w;
  w.factory("Gaussian::g(x[-5,5],mu[0,-3,3],sigma[1])");
  auto x = w.var("x");
  RooAbsPdf *pdf = w.pdf("g");
  RooDataSet *data = pdf->generate(RooArgSet(*x), 10000);

  int mpfe_task_mode = 0;
  if (mp_task_mode == RooNLLVarTask::interleave) {
    mpfe_task_mode = 1;
  }

  auto nll_mpfe = pdf->createNLL(*data, RooFit::NumCPU(NumCPU, mpfe_task_mode));

  auto mpfe_result = nll_mpfe->getVal();

  // create new nll without MPFE for creating nll_mp (an MPFE-enabled RooNLLVar interferes with MP::Vector's bipe use)
  auto nll = pdf->createNLL(*data);
  MPRooNLLVar nll_mp(NumCPU, mp_task_mode, *dynamic_cast<RooNLLVar*>(nll));

  auto mp_result = nll_mp.getVal();

  EXPECT_EQ(Hex(mpfe_result), Hex(mp_result));
  if (HasFailure()) {
    std::cout << "failed test had parameters NumCPU = " << NumCPU << ", task_mode = " << mp_task_mode << ", seed = " << seed << std::endl;
  }
}


TEST_P(NLLMultiProcessVsMPFE, minimize) {
  // do a minimization (e.g. like in GradMinimizer_Gaussian1D test)

  // TODO: see whether it performs adequately

  // parameters
  std::size_t NumCPU = std::get<0>(GetParam());
  RooNLLVarTask mp_task_mode = std::get<1>(GetParam());
  std::size_t seed = std::get<2>(GetParam());

  RooRandom::randomGenerator()->SetSeed(seed);

  RooWorkspace w = RooWorkspace();

  w.factory("Gaussian::g(x[-5,5],mu[0,-3,3],sigma[1])");
  auto x = w.var("x");
  RooAbsPdf *pdf = w.pdf("g");
  RooRealVar *mu = w.var("mu");

  RooDataSet *data = pdf->generate(RooArgSet(*x), 10000);
  mu->setVal(-2.9);

  int mpfe_task_mode;
  switch (mp_task_mode) {
    case RooNLLVarTask::bulk_partition: {
      mpfe_task_mode = 0;
      break;
    }
    case RooNLLVarTask::interleave: {
      mpfe_task_mode = 1;
      break;
    }
    default: {
      throw std::logic_error("can only compare bulk_partition and interleave strategies to MPFE NLL");
    }
  }

  auto nll_mpfe = pdf->createNLL(*data, RooFit::NumCPU(NumCPU, mpfe_task_mode));
  auto nll_nominal = pdf->createNLL(*data);
  MPRooNLLVar nll_mp(NumCPU, mp_task_mode, *dynamic_cast<RooNLLVar*>(nll_nominal));

  // save initial values for the start of all minimizations
  RooArgSet values = RooArgSet(*mu, *pdf);

  RooArgSet *savedValues = dynamic_cast<RooArgSet *>(values.snapshot());
  if (savedValues == nullptr) {
    throw std::runtime_error("params->snapshot() cannot be casted to RooArgSet!");
  }

  // --------

  RooMinimizer m0(*nll_mpfe);
  m0.setMinimizerType("Minuit2");

  m0.setStrategy(0);
  m0.setPrintLevel(-1);

  m0.migrad();

  RooFitResult *m0result = m0.lastMinuitFit();
  double minNll0 = m0result->minNll();
  double edm0 = m0result->edm();
  double mu0 = mu->getVal();
  double muerr0 = mu->getError();

  values = *savedValues;

  RooMinimizer m1(nll_mp);
  m1.setMinimizerType("Minuit2");

  m1.setStrategy(0);
  m1.setPrintLevel(-1);

  m1.migrad();

  RooFitResult *m1result = m1.lastMinuitFit();
  double minNll1 = m1result->minNll();
  double edm1 = m1result->edm();
  double mu1 = mu->getVal();
  double muerr1 = mu->getError();

  EXPECT_EQ(minNll0, minNll1);
  EXPECT_EQ(mu0, mu1);
  EXPECT_EQ(muerr0, muerr1);
  EXPECT_EQ(edm0, edm1);
}


INSTANTIATE_TEST_CASE_P(NworkersModeSeed,
                        NLLMultiProcessVsMPFE,
                        ::testing::Combine(::testing::Values(2,3,4),  // number of workers
                                           ::testing::Values(RooNLLVarTask::bulk_partition,
                                                             RooNLLVarTask::interleave),
                                           ::testing::Values(2,3)));  // random seed


TEST(NLLMultiProcessVsMPFE, throwOnCreatingMPwithMPFE) {
  // Using an MPFE-enabled NLL should throw when creating an MP NLL.
  RooWorkspace w;
  w.factory("Gaussian::g(x[-5,5],mu[0,-3,3],sigma[1])");
  auto x = w.var("x");
  RooAbsPdf *pdf = w.pdf("g");
  RooDataSet *data = pdf->generate(RooArgSet(*x), 10);

  auto nll_mpfe = pdf->createNLL(*data, RooFit::NumCPU(2));

  EXPECT_THROW({
    MPRooNLLVar nll_mp(2, RooNLLVarTask::bulk_partition, *dynamic_cast<RooNLLVar*>(nll_mpfe));
  }, std::logic_error);
}


