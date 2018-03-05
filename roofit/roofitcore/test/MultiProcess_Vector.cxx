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
#include <queue>
#include <exception>

#include <RooRealVar.h>
#include <../src/BidirMMapPipe.h>
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

    // IPCQ (InterProcess Communication and Queue) handles message passing
    // and communication with a queue of tasks and workers that execute the
    // tasks. The queue is in a separate process that can communicate with the
    // master process (from where this object is created) and the queue process
    // communicates with the worker processes.
    //
    // The IPCQ class is templated upon a class that defines the actual
    // work that must be done, Job (think of "job" as in "employment", e.g. the
    // job of a baker, which involves *tasks* like baking and selling bread).
    // This class must implement:
    // - Job::task_type, the type of task that is passed to and from the queue.
    // - Job::process_message_queue(int), a function on the queue process that
    //   receives an integer message and performs the corresponding action.
    // - Job::process_message_worker(int), same, but on a worker process.
    //
    // For message passing, any type of message T can be sent. The implementer
    // must make sure that T can be sent over the BidirMMapPipe, i.e. that
    // BidirMMapPipe::operator<<(T) and >>(T) are implemented. This can be done
    // locally in the implementation file of the messenger, e.g.:
    //
    //    namespace RooFit {
    //      struct T {
    //        int a, b;
    //      };
    //      BidirMMapPipe& BidirMMapPipe::operator<<(const T & sent) {
    //        *this << sent.a << sent.b;
    //        return *this;
    //      }
    //      BidirMMapPipe& BidirMMapPipe::operator>>(T & received) {
    //        *this >> received.a >> received.b;
    //        return *this;
    //      }
    //    }
    //
    // Make sure that activate() is called soon after creation of an IPCQ,
    // because everything in between construction and activate() gets executed
    // both on the master process and on the slaves. Activate starts the queue
    // loop on the queue process and the work loop on the workers, which means
    // they can start doing their jobs. Activate cannot be called from inside
    // the constructor, since the loops would prevent the constructor from
    // returning a constructed object (thus defying its purpose).


    template <class Job>
    class InterProcessQueueAndMessenger {
     public:
      // constructor
      explicit InterProcessQueueAndMessenger(std::size_t NumCPU) {
        // first fork queuing process (done in initialization), then workers from that
        if (queue_pipe.isChild()) {
          _is_master = false;
          for (std::size_t ix = 0; ix < NumCPU; ++ix) {
            // set worker_id before each fork so that fork will sync it to the worker
            worker_id = ix;
            worker_pipes.emplace_back();
          }
          if (worker_pipes.back().isParent()) {
            _is_queue = true;
          }
        }
      }

      // start message loops on slave processes
      void activate() {
        // should be called soon after creation of this object, because everything in
        // between construction and activate gets executed both on the master process
        // and on the slaves
        if (_is_master) {
          // on master we only have to set the activated flag so that we can start
          // queuing tasks
          activated = true;
          // it's not important on the other processes
        } else if (_is_queue) {
          queue_loop();
        } else { // is worker
          worker_loop();
        }
      }

      void worker_loop() {
        bool carry_on = true;
        T_task task_index;
        int message;
        while (carry_on) {
          if (from_master_queue(task_index)) {
            evaluate_task(task_index);
          } else if (queuemunicator->from_master(message)) {
            if (message == 0) {
              // get a task from the queue
            } else if (message == -1) {
              // terminate
              carry_on = false;
            } else {
              process_message(message);
            }
          }
        }
      }


      // Send a message vector from master to all slaves
      template <typename T_message>
      void to_slaves(T_message message) {
        for (const RooFit::BidirMMapPipe & pipe : worker_pipes) {
          if (pipe.isParent()) {
            pipe << message;
          } else {
            throw std::logic_error("calling Communicator::to_slaves from slave process");
          }
        }
      }


      template <typename T_message>
      bool from_master(T_message & message) {
        RooFit::BidirMMapPipe& pipe = worker_pipes[worker_id];
        if (pipe.isChild()) {
          pipe >> message;
        } else {
          throw std::logic_error("calling Communicator::from_master from master process");
        }
      }


      // Send a message from a slave back to master
      template <typename T_message>
      void to_master(T_message message) {
        RooFit::BidirMMapPipe& pipe = worker_pipes[worker_id];
        if (pipe.isChild()) {
          pipe << message;
        } else {
          throw std::logic_error("calling Communicator::to_master from master process");
        }
      }


      // Have a worker ask for a task-message from the queue
      bool from_queue(T_task & task) {
        if (queue.empty()) {
          return false;
        } else {
          task = queue.pop();
          return true;
        }
      }


      // Enqueue a task
      void to_queue(T_task task) {
        if (is_master()) {
          if (!activated) {
            activate();
          }

          queue.push(task);

        } else {
          throw std::logic_error("calling Communicator::to_master_queue from slave process");
        }
      }


      bool is_master() {
        return _is_master;
      }


     private:
      std::vector<RooFit::BidirMMapPipe> worker_pipes;
      RooFit::BidirMMapPipe queue_pipe;
      std::size_t worker_id;
      bool _is_master = true;
      bool _is_queue = false;
      std::queue<T_task> queue;
      bool activated = false;
    };




    // Vector defines an interface and communication machinery to build a
    // parallelized subclass of an existing non-concurrent numerical class that
    // can be expressed as a vector of independent sub-calculations.
    //
    // A subclass can communicate between master and worker processes using
    // messages that the subclass defines for itself. The interface uses int
    // type for messages. Two messages are predefined:
    // * -1 means terminate the worker_loop
    // *  0 means take a new task from the queue (essentially: no message)
    // * any other value is deferred to the subclass-defined process_message
    //   method.
    //
    template <typename Base, typename T_task = std::size_t>
    class Vector : public Base {
     public:
      template <typename... Targs>
      Vector(std::size_t NumCPU, Targs ...args) :
          Base(args...),
          _NumCPU(NumCPU)
      {
//        task_indices.reserve(num_tasks_from_cpus());
//        for (std::size_t ix = 0; ix < num_tasks_from_cpus(); ++ix) {
//          task_indices.emplace_back(ix);
//        }
      }

      ~Vector() {
        delete queuemunicator;
      }

      void initialize_parallel_work_system() {
        if (queuemunicator == nullptr) {
          queuemunicator = new Queuemunicator<T_task>(_NumCPU);
        }
        if (!queuemunicator->is_master()) {
          worker_loop();
        }
      }

     private:
      virtual void evaluate_task(T_task task_index) = 0;
      virtual void sync_worker(std::size_t worker_id) {};


      virtual void process_message(int m) = 0;

      virtual std::size_t num_tasks_from_cpus() {
        return 1;
      };

      void worker_loop() {
        bool carry_on = true;
        T_task task_index;
        int message;
        while (carry_on) {
          if (queuemunicator->from_master_queue(task_index)) {
            evaluate_task(task_index);
          } else if (queuemunicator->from_master(message)) {
            if (message == 0) {
              // get a task from the queue
            } else if (message == -1) {
              // terminate
              carry_on = false;
            } else {
              process_message(message);
            }
          }
        }
      }

     protected:
//      std::vector<std::size_t> task_indices;
      std::size_t _NumCPU;
      IPCQ<T_task> * queuemunicator = nullptr;

      template <typename T>
      void enqueue_message(int m, T a) {};
      template <typename T1, typename T2>
      void enqueue_message(int m, T1 a1, T2 a2) {};
    };

  }
}


class xSquaredPlusBVectorParallel : public RooFit::MultiProcess::Vector<xSquaredPlusBVectorSerial> {
  using BASE = RooFit::MultiProcess::Vector<xSquaredPlusBVectorSerial>;
 public:
  xSquaredPlusBVectorParallel(std::size_t NumCPU, double b_init, std::vector<double> x_init) :
      BASE(NumCPU, b_init, x_init) // NumCPU stands for everything that defines the parallelization behaviour (number of cpu, strategy, affinity etc)
  {}

  enum class Message : int {set_b_worker = 1, evaluate_tasks = 2, retrieve_task_elements = 3};

  void evaluate() override {
//    // sync remote first: local b -> workers
//    sync();
//    // choose parallel strategy from multiprocess vector


    // master fills queue with tasks
    for (std::size_t ix = 0; ix < x.size(); ++ix) {
      queuemunicator->to_master_queue(ix);
    }
    // wait for workers to evaluate tasks; this should happen automatically if no messages are being processed... but how? do we need a queue_loop? then we would also need a separate queue process, because otherwise it won't be asynchronous with the master...
//    queuemunicator->to_slaves(Message::evaluate_tasks);
    // wait for task results back from worker to master
    result = queuemunicator->get
  }

  void set_b_workers(double b) {}


  void sync() {
    // implementation defines sync, in this case update b only
//    for (auto worker_id : workers) {
//      sync_worker(worker_id);
//    }
  }


 private:
  // Gets called from the server loop (on worker) when a message arrives.
  // Internally forwards to an overload with a Message class enum, which is
  // just for convenience, not a requirement.
  void process_message(int m) override {
    process_message(static_cast<Message>(m));
  }

  void process_message(Message m) {
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
      case Message::evaluate_tasks: {
        evaluate_task()
        break;
      }
      default: {
        throw std::runtime_error("go away");
      }
    }
  }

  using BASE::enqueue_message;  // to unhide the method from the base class that we're going to overload next

  template <typename... Targs>
  void enqueue_message(Message m, Targs ...args) {
    enqueue_message(static_cast<int>(m), args...);
  }

  void sync_worker(std::size_t worker_id) override {
    enqueue_message(Message::set_b_worker, _b.getVal());
//    set_b_worker(_b, worker_id);
  }

  void evaluate_task(std::size_t task_index) {
    result[task_index] = std::pow(x[task_index], 2) + _b.getVal();
  } // if serial implementation doesn't define evaluate_task -> implement here

  virtual std::size_t num_tasks_from_cpus() {
    return _NumCPU;
  }

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
  x_sq_plus_b_parallel.initialize_parallel_work_system();

  auto y_parallel = x_sq_plus_b_parallel.get_result();

  EXPECT_EQ(y_parallel[0], y_expected[0]);
  EXPECT_EQ(y_parallel[1], y_expected[1]);
  EXPECT_EQ(y_parallel[2], y_expected[2]);
  EXPECT_EQ(y_parallel[3], y_expected[3]);
}
