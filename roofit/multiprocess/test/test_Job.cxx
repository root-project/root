/*
 * Project: RooFit
 * Authors:
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl
 *   IP, Inti Pelupessy, Netherlands eScience Center, i.pelupessy@esciencecenter.nl
 *   VC, Vince Croft, DIANA / NYU, vincent.croft@cern.ch
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include <cmath>
#include <vector>

#include "RooFit/MultiProcess/Job.h"
#include "RooFit/MultiProcess/types.h" // JobTask
#include "RooFit/MultiProcess/Config.h"
// needed to complete type returned from...
#include "RooFit/MultiProcess/JobManager.h"     // ... Job::get_manager()
#include "RooFit/MultiProcess/ProcessManager.h" // ... JobManager::process_manager()
#include "RooFit/MultiProcess/Queue.h"          // ... JobManager::queue()

#include "gtest/gtest.h"
#include "utils.h"

class xSquaredPlusBVectorSerial {
public:
   xSquaredPlusBVectorSerial(double b, const std::vector<double> &x) : b_(b), x_(x), result_(x.size()) {}

   void evaluate()
   {
      // call evaluate_task for each task
      for (std::size_t ix = 0; ix < x_.size(); ++ix) {
         result_[ix] = std::pow(x_[ix], 2) + b_;
      }
   }

   std::vector<double> get_result()
   {
      evaluate();
      return result_;
   }

   // for simplicity of the example (avoiding getters/setters) we make data members public as well
   double b_;
   std::vector<double> x_;
   std::vector<double> result_;
};

class xSquaredPlusBVectorParallel : public RooFit::MultiProcess::Job {
public:
   explicit xSquaredPlusBVectorParallel(xSquaredPlusBVectorSerial *serial, bool update_state = false)
      : serial_(serial), update_state_(update_state)
   {
   }

   void evaluate()
   {
      if (get_manager()->process_manager().is_master()) {
         if (update_state_) {
            update_state();
         }
         // master fills queue with tasks
         for (std::size_t task_id = 0; task_id < serial_->x_.size(); ++task_id) {
            RooFit::MultiProcess::JobTask job_task{id_, state_id_, task_id};
            get_manager()->queue().add(job_task);
            ++N_tasks_at_workers_;
         }

         // wait for task results back from workers to master
         gather_worker_results();
      }
   }

   std::vector<double> get_result()
   {
      evaluate();
      return serial_->result_;
   }

   // -- BEGIN plumbing --

   // typically, on master, this would be called inside evaluate, before queuing tasks; on workers it's called
   // automatically when a published message is received from master
   void update_state() override
   {
      if (get_manager()->process_manager().is_master()) {
         ++state_id_;
         get_manager()->messenger().publish_from_master_to_workers(
            id_, state_id_, serial_->b_); // always send Job id first! This is used in worker_loop to route the
                                          // update_state call to the correct Job.
      } else if (get_manager()->process_manager().is_worker()) {
         state_id_ = get_manager()->messenger().receive_from_master_on_worker<RooFit::MultiProcess::State>();
         serial_->b_ = get_manager()->messenger().receive_from_master_on_worker<double>();
      }
   }

   struct task_result_t {
      std::size_t job_id; // job ID must always be the first part of any result message/type
      std::size_t task_id;
      double value;
   };

   void send_back_task_result_from_worker(std::size_t task) override
   {
      task_result_t task_result{id_, task, serial_->result_[task]};
      zmq::message_t message(sizeof(task_result_t));
      memcpy(message.data(), &task_result, sizeof(task_result_t));
      get_manager()->messenger().send_from_worker_to_master(std::move(message));
   }

   bool receive_task_result_on_master(const zmq::message_t &message) override
   {
      auto result = message.data<task_result_t>();
      serial_->result_[result->task_id] = result->value;
      --N_tasks_at_workers_;
      bool job_completed = (N_tasks_at_workers_ == 0);
      return job_completed;
   }

   // -- END plumbing --

private:
   void evaluate_task(std::size_t task) override
   {
      assert(get_manager()->process_manager().is_worker());
      serial_->result_[task] = std::pow(serial_->x_[task], 2) + serial_->b_;
   }

   xSquaredPlusBVectorSerial *serial_;
   bool update_state_ = false;
   std::size_t N_tasks_at_workers_ = 0;
};

class TestMPJob : public ::testing::TestWithParam<std::size_t> {
   // You can implement all the usual fixture class members here.
   // To access the test parameter, call GetParam() from class
   // TestWithParam<T>.
};

TEST_P(TestMPJob, singleJobGetResult)
{
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

   xSquaredPlusBVectorParallel x_sq_plus_b_parallel(&x_sq_plus_b);
   RooFit::MultiProcess::Config::setDefaultNWorkers(NumCPU);

   auto y_parallel = x_sq_plus_b_parallel.get_result();

   EXPECT_EQ(Hex(y_parallel[0]), Hex(y_expected[0]));
   EXPECT_EQ(Hex(y_parallel[1]), Hex(y_expected[1]));
   EXPECT_EQ(Hex(y_parallel[2]), Hex(y_expected[2]));
   EXPECT_EQ(Hex(y_parallel[3]), Hex(y_expected[3]));
}

TEST_P(TestMPJob, multiJobGetResult)
{
   // Simple test case: calculate x^2 + b, where x is a vector. This case does
   // both a simple calculation (squaring the input vector x) and represents
   // handling of state updates in b.
   std::vector<double> x{0, 1, 2, 3};
   double b_initial = 3.;

   std::vector<double> y_expected{3, 4, 7, 12};

   std::size_t NumCPU = GetParam();

   // define jobs
   xSquaredPlusBVectorSerial x_sq_plus_b(b_initial, x);
   xSquaredPlusBVectorSerial x_sq_plus_b2(b_initial + 1, x);
   xSquaredPlusBVectorParallel x_sq_plus_b_parallel(&x_sq_plus_b);
   xSquaredPlusBVectorParallel x_sq_plus_b_parallel2(&x_sq_plus_b2);
   RooFit::MultiProcess::Config::setDefaultNWorkers(NumCPU);

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

TEST_P(TestMPJob, singleJobUpdateState)
{
   // Simple test case: calculate x^2 + b, where x is a vector. This case does
   // both a simple calculation (squaring the input vector x) and represents
   // handling of state updates in b.
   std::vector<double> x{0, 1, 2, 3};
   double b_initial = 1.;
   xSquaredPlusBVectorSerial x_sq_plus_b(b_initial, x);

   std::vector<double> y_expected{3, 4, 7, 12};

   std::size_t NumCPU = GetParam();

   // start parallel test
   bool update_state = true;
   xSquaredPlusBVectorParallel x_sq_plus_b_parallel(&x_sq_plus_b, update_state);
   RooFit::MultiProcess::Config::setDefaultNWorkers(NumCPU);

   auto y_parallel_before_change = x_sq_plus_b_parallel.get_result();

   x_sq_plus_b.b_ = 3.;

   auto y_parallel_after_change = x_sq_plus_b_parallel.get_result();

   EXPECT_NE(Hex(y_parallel_before_change[0]), Hex(y_expected[0]));
   EXPECT_NE(Hex(y_parallel_before_change[1]), Hex(y_expected[1]));
   EXPECT_NE(Hex(y_parallel_before_change[2]), Hex(y_expected[2]));
   EXPECT_NE(Hex(y_parallel_before_change[3]), Hex(y_expected[3]));

   EXPECT_EQ(Hex(y_parallel_after_change[0]), Hex(y_expected[0]));
   EXPECT_EQ(Hex(y_parallel_after_change[1]), Hex(y_expected[1]));
   EXPECT_EQ(Hex(y_parallel_after_change[2]), Hex(y_expected[2]));
   EXPECT_EQ(Hex(y_parallel_after_change[3]), Hex(y_expected[3]));
}

INSTANTIATE_TEST_SUITE_P(NumberOfWorkerProcesses, TestMPJob, ::testing::Values(1, 2, 3));
