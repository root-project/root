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

#include <cstdlib>  // std::_Exit
#include <cmath>
#include <vector>
#include <queue>
#include <map>
#include <exception>
#include <type_traits>  // is_enum, is_convertible

#include <RooRealVar.h>
#include <../src/BidirMMapPipe.h>
//#include <roofit/MultiProcess/Vector.h>

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


namespace RooFit {
  namespace MultiProcess {

    // Messages from master to queue
    enum class M2Q : int {
      terminate = -1,
      enqueue = 0,
      retrieve = 1
    };

    // Messages from queue to master
    enum class Q2M : int {
      retrieve_rejected = 0,
      retrieve_accepted = 1
    };


    // Messages from worker to queue
    enum class W2Q : int {
      dequeue = 0,
      send_result = 1
    };

    // Messages from queue to worker
    enum class Q2W : int {
      dequeue_rejected = 0,
      dequeue_accepted = 1,
      update_parameter = 2,
      switch_work_mode = 3
    };
  }
}

// stream operators for message enum classes
namespace RooFit {

//  namespace detail {
//    template<typename T>
//    using is_class_enum = std::integral_constant<
//        bool,
//        std::is_enum<T>::value && !std::is_convertible<T, int>::value>;
//
//    template<typename T>
//    using is_class_enum_v = is_class_enum<T>::value;
//
//    template<bool B, class T = void>
//    using enable_if_t = typename std::enable_if<B, T>::type;
//
//    template<typename E>
//    using enable_if_is_class_enum_t = enable_if_t<is_class_enum_v<E>, E>;
//  }
//
//  template<typename E>
//  BidirMMapPipe &BidirMMapPipe::operator<<(const detail::enable_if_is_class_enum_t<E> &sent) {
//    *this << static_cast<typename std::underlying_type<E>::type>(sent);
//    return *this;
//  }
//
//  template<typename E>
//  BidirMMapPipe &BidirMMapPipe::operator>>(detail::enable_if_is_class_enum_t<E> &received) {
//    typename std::underlying_type<E>::type receptor;
//    *this >> receptor;
//    received = static_cast<E>(receptor);
//    return *this;
//  }

  BidirMMapPipe &BidirMMapPipe::operator<<(const MultiProcess::M2Q& sent) {
    *this << static_cast<int>(sent);
    return *this;
  }

  BidirMMapPipe &BidirMMapPipe::operator>>(MultiProcess::M2Q& received) {
    int receptor;
    *this >> receptor;
    received = static_cast<MultiProcess::M2Q>(receptor);
    return *this;
  }

  BidirMMapPipe &BidirMMapPipe::operator<<(const MultiProcess::Q2M& sent) {
    *this << static_cast<int>(sent);
    return *this;
  }

  BidirMMapPipe &BidirMMapPipe::operator>>(MultiProcess::Q2M& received) {
    int receptor;
    *this >> receptor;
    received = static_cast<MultiProcess::Q2M>(receptor);
    return *this;
  }

  BidirMMapPipe &BidirMMapPipe::operator<<(const MultiProcess::W2Q& sent) {
    *this << static_cast<int>(sent);
    return *this;
  }

  BidirMMapPipe &BidirMMapPipe::operator>>(MultiProcess::W2Q& received) {
    int receptor;
    *this >> receptor;
    received = static_cast<MultiProcess::W2Q>(receptor);
    return *this;
  }

  BidirMMapPipe &BidirMMapPipe::operator<<(const MultiProcess::Q2W& sent) {
    *this << static_cast<int>(sent);
    return *this;
  }

  BidirMMapPipe &BidirMMapPipe::operator>>(MultiProcess::Q2W& received) {
    int receptor;
    *this >> receptor;
    received = static_cast<MultiProcess::Q2W>(receptor);
    return *this;
  }

}


namespace RooFit {
  namespace MultiProcess {

    class Job {
     public:
      virtual void evaluate_task(std::size_t task) = 0;
      virtual double get_task_result(std::size_t task) = 0;
    };


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
    // - Job::Task, the type of a task identifier that is passed to and from
    //   the queue.
    // - Job::Result, the type of task results that is eventually passed from
    //   the workers back to the master process. These have to go through the
    //   queue process, so the queue needs to know their type.
    // - Job::process_message(int), a function on a worker process that
    //   receives an integer message and performs the corresponding action.
    // - Job::worker_loop(), receives messages and handles them. IPCQ expects:
    //   * The loop to continue indefinitely, unless told otherwise.
    //   * The loop is exited on message -1, but not before sending -1 back as
    //     a handshake message.
    //   * When receiving message 0 from worker, IPCQ will send:
    //     + message 0 if the queue is empty
    //     + message 1 and a task from the queue if the queue is not empty.
    //     The worker_loop should wait for this when sending 0.
    //   * When receiving message 1 from worker, IPCQ will expect a task
    //     identifier followed by a result from that task.
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
    // loop on the queue process, which means it can start doing its job.
    // Worker processes have to be activated separately from the Job objects
    // themselves. Activate cannot be called from inside the constructor, since
    // the loops would prevent the constructor from returning a constructed
    // object (thus defying its purpose). Note that at the end of activate, the
    // child processes are killed. This is achieved by sending the terminate
    // message, which is done automatically in the destructor of this class,
    // but can also be done manually via terminate().

    using Task = std::size_t;
    using JobTask = std::pair<std::size_t, Task>;  // combined job_object and task identifier type

//    template <class Task, class Result>
    class InterProcessQueueAndMessenger {

     public:
      static InterProcessQueueAndMessenger& instance(std::size_t NumCPU) {
        if (_instance == nullptr) {
          assert(NumCPU != 0);
          _instance = new InterProcessQueueAndMessenger(NumCPU);
        } else {
          assert(NumCPU == _instance->worker_pipes.size());
        }
        return *_instance;
      }

      static InterProcessQueueAndMessenger& instance() {
        assert(_instance != nullptr);
        return *_instance;
      }

      // constructor
     private:
      explicit InterProcessQueueAndMessenger(std::size_t NumCPU) {
        // first fork queuing process (done in initialization), then workers from that
        if (queue_pipe.isChild()) {
          _is_master = false;
          // reserve is necessary! BidirMMapPipe is not allowed to be copied,
          // but when capacity is not enough when using emplace_back, the
          // vector must be resized, which means existing elements must be
          // copied to the new memory locations.
          worker_pipes.reserve(NumCPU);
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


     public:
      ~InterProcessQueueAndMessenger() {
        terminate();
      }


      static std::size_t add_job_object(Job *job_object) {
        job_objects.push_back(job_object);
        return job_objects.size() - 1;
      }

      static Job* get_job_object(std::size_t job_object_id) {
        return job_objects[job_object_id];
      }


      // protocol for terminating processes: send message -1 and wait for child
      // to send back the same message -1.
      static void terminate_pipe(BidirMMapPipe &pipe, std::string error_message) {
        pipe << -1;
        // wait for handshake:
        int message;
        pipe >> message;
        if (message != -1 || 0 != pipe.close()) {
          std::cerr << error_message << std::endl;
        }
      }


      void terminate() {
        if (_is_master) {
          terminate_pipe(queue_pipe, "In terminate: queue shutdown failed.");
        }
      }


      void terminate_workers() {
        if (_is_queue) {
          for (BidirMMapPipe &worker_pipe : worker_pipes) {
            terminate_pipe(worker_pipe, "In terminate_workers: worker shutdown failed.");
          }
        }
      }


      // start message loops on child processes and quit processes afterwards
      void activate() {
        // should be called soon after creation of this object, because everything in
        // between construction and activate gets executed both on the master process
        // and on the slaves
        if (_is_master) {
          // on master we only have to set the activated flag so that we can start
          // queuing tasks
          queue_activated = true;
          // it's not important on the other processes
        } else if (_is_queue) {
          queue_loop();
          std::_Exit(0);
        } else { // is worker
          queue_activated = true;
        }
      }


      BidirMMapPipe::PollVector get_poll_vector() {
        BidirMMapPipe::PollVector poll_vector;
        poll_vector.reserve(1 + worker_pipes.size());
        poll_vector.emplace_back(&queue_pipe);
        for (BidirMMapPipe &pipe : worker_pipes) {
          poll_vector.emplace_back(&pipe);
        }
        return poll_vector;
      }


      bool process_queue_pipe_message(int message) {
        bool carry_on = true;

        switch (message) {
          case -1: {
            // terminate
            // pass on the signal to workers:
            terminate_workers();
            // stop queue-loop on next iteration:
            carry_on = false;
            break;
          }

          case 0: {
            // enqueue task
            Task task;
            std::size_t job_object_id;
            queue_pipe >> job_object_id >> task;
            JobTask job_task(job_object_id, task);
            to_queue(job_task);
            N_tasks++;
            break;
          }

          case 1: {
            // retrieve task results after queue is empty and all
            // tasks have been completed
            if (queue.empty() && results.size() == N_tasks) {
              queue_pipe << Q2M::retrieve_accepted;  // handshake message (master will now start reading from the pipe)
              queue_pipe << N_tasks;
              for (auto const &item : results) {
                queue_pipe << item.first.first << item.first.second << item.second;
              }
              // empty results cache
              results.clear();
              // reset number of received tasks
              N_tasks = 0;
            } else {
              queue_pipe << Q2M::retrieve_rejected;  // handshake message: tasks not done yet, try again
            }
          }
        }

        return carry_on;
      }


      void retrieve() {
        if (_is_master) {
          bool carry_on = true;
          while (carry_on) {
            queue_pipe << M2Q::retrieve;
            Q2M handshake;
            queue_pipe >> handshake;
            if (handshake == Q2M::retrieve_accepted) {
              carry_on = false;
              queue_pipe >> N_tasks;
              for (std::size_t ix = 0; ix < N_tasks; ++ix) {
                Task task;
                std::size_t job_object_id;
                double result;
                queue_pipe >> job_object_id >> task >> result;
                JobTask job_task(job_object_id, task);
                results[job_task] = result;
              }
            }
          }
        }
      }


      void process_worker_pipe_message(BidirMMapPipe &pipe, int message) {
        Task task;
        std::size_t job_object_id;
        switch (message) {
          case 0: {
            // dequeue task
            JobTask job_task;
            if (from_queue(job_task)) {
              pipe << Q2W::dequeue_accepted << job_task.first << job_task.second;
            } else {
              pipe << Q2W::dequeue_rejected;
            }
            break;
          }

          case 1: {
            // receive back task result and store it for later
            // sending back to master
            // TODO: add RooAbsCategory handling!
            double result;
            pipe >> job_object_id >> task >> result;
            JobTask job_task(job_object_id, task);
            results[job_task] = result;
          }
        }
      }


      void queue_loop() {
        if (_is_queue) {
          bool carry_on = true;
          auto poll_vector = get_poll_vector();

          while (carry_on) {
            // poll: wait until status change (-1: infinite timeout)
            int n_changed_pipes = BidirMMapPipe::poll(poll_vector, -1);
            // scan for pipes with changed status:
            for (auto it = poll_vector.begin(); n_changed_pipes > 0 && poll_vector.end() != it; ++it) {
              --n_changed_pipes; // maybe we can stop early...
              // read from pipes which are readable
              if (it->revents & BidirMMapPipe::Readable) {
                int message;
                BidirMMapPipe &pipe = *(it->pipe);

                pipe >> message;

                // message comes from the master/queue pipe (which is the first
                // element):
                if (it == poll_vector.begin()) {
                  carry_on = process_queue_pipe_message(message);
                  // on terminate, also stop for-loop, no need to check other
                  // pipes anymore:
                  if (!carry_on) {
                    n_changed_pipes = 0;
                  }
                } else { // from a worker pipe
                  process_worker_pipe_message(pipe, message);
                }
              }
            }
          }
        }
      }


      // Send a message vector from master to all slaves
      template<typename T_message>
      void to_slaves(T_message message) {
        for (const RooFit::BidirMMapPipe &pipe : worker_pipes) {
          if (pipe.isParent()) {
            pipe << message;
          } else {
            throw std::logic_error("calling Communicator::to_slaves from slave process");
          }
        }
      }


      template<typename T_message>
      bool from_master(T_message &message) {
        RooFit::BidirMMapPipe &pipe = worker_pipes[worker_id];
        if (pipe.isChild()) {
          pipe >> message;
        } else {
          throw std::logic_error("calling Communicator::from_master from master process");
        }
      }


      // Send a message from a slave back to master
      template<typename T_message>
      void to_master(T_message message) {
        RooFit::BidirMMapPipe &pipe = worker_pipes[worker_id];
        if (pipe.isChild()) {
          pipe << message;
        } else {
          throw std::logic_error("calling Communicator::to_master from master process");
        }
      }


      // Have a worker ask for a task-message from the queue
      bool from_queue(JobTask &job_task) {
        if (queue.empty()) {
          return false;
        } else {
          job_task = queue.front();
          queue.pop();
          return true;
        }
      }


      // Enqueue a task
      void to_queue(JobTask job_task) {
        if (is_master()) {
          if (!queue_activated) {
            activate();
          }
          queue_pipe << 1 << job_task.first << job_task.second;

        } else if (is_queue()) {
          queue.push(job_task);
        } else {
          throw std::logic_error("calling Communicator::to_master_queue from slave process");
        }
      }


      bool is_master() {
        return _is_master;
      }

      bool is_queue() {
        return _is_queue;
      }

      bool is_worker() {
        return !(_is_master || _is_queue);
      }


      RooFit::BidirMMapPipe& get_worker_pipe() {
        assert(is_worker());
        return worker_pipes[worker_id];
      }


      std::map<JobTask, double>& get_results() {
        return results;
      }

     public:



     private:
      std::vector<RooFit::BidirMMapPipe> worker_pipes;
      RooFit::BidirMMapPipe queue_pipe;
      std::size_t worker_id;
      bool _is_master = true;
      bool _is_queue = false;
      std::queue<JobTask> queue;
      std::size_t N_tasks = 0;  // total number of received tasks
      // TODO: add RooAbsCategory handling to results!
      std::map<JobTask, double> results;
      bool queue_activated = false;

      static std::vector<Job *> job_objects;
      static InterProcessQueueAndMessenger *_instance;
    };

    // initialize static members
    std::vector<Job *> InterProcessQueueAndMessenger::job_objects;
    InterProcessQueueAndMessenger * InterProcessQueueAndMessenger::_instance = nullptr;

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
    template <typename Base>
    class Vector : public Base, public Job {
     public:
      template<typename... Targs>
      Vector(std::size_t NumCPU, Targs ...args) :
          Base(args...),
          _NumCPU(NumCPU) {
        job_id = InterProcessQueueAndMessenger::add_job_object(this);
//        task_indices.reserve(num_tasks_from_cpus());
//        for (std::size_t ix = 0; ix < num_tasks_from_cpus(); ++ix) {
//          task_indices.emplace_back(ix);
//        }
      }

      ~Vector() {
        delete ipqm;
      }

      void initialize_parallel_work_system() {
        if (ipqm == nullptr) {
          ipqm = &InterProcessQueueAndMessenger::instance(_NumCPU);
        }

        ipqm->activate();

        if (ipqm->is_worker()) {
          worker_loop();
        }
      }

      static void worker_loop() {
        assert(InterProcessQueueAndMessenger::instance().is_worker());
        bool carry_on = true;
        Task task;
        std::size_t job_object_id;
        Q2W message_q2w;
        JobTask job_task;
        BidirMMapPipe& pipe = InterProcessQueueAndMessenger::instance().get_worker_pipe();
        while (carry_on) {
          if (work_mode) {
            // try to dequeue a task
            pipe << W2Q::dequeue;
            // receive handshake
            pipe >> message_q2w;

            switch (message_q2w) {
              case Q2W::dequeue_rejected: {
                break;
              }
              case Q2W::dequeue_accepted: {
                pipe >> job_object_id >> task;
                InterProcessQueueAndMessenger::get_job_object(job_object_id)->evaluate_task(task);

                // TODO: add RooAbsCategory handling!
                double result = InterProcessQueueAndMessenger::get_job_object(job_object_id)->get_task_result(task);
                pipe << W2Q::send_result << job_object_id << task << result;

                break;
              }

              case Q2W::switch_work_mode: {
                // change to non-work-mode
                work_mode = false;
                break;
              }

              case Q2W::update_parameter: {
                std::cerr << "In worker_loop: update_parameter message invalid in work-mode!" << std::endl;
              }
            }
          } else {
            // receive message
            pipe >> message_q2w;

            switch (message_q2w) {
              case Q2W::update_parameter: {
                // receive new parameter value and update
                // ...
                break;
              }

              case Q2W::switch_work_mode: {
                // change to work-mode
                work_mode = true;
                break;
              }

              case Q2W::dequeue_accepted:
              case Q2W::dequeue_rejected: {
                std::cerr << "In worker_loop: dequeue_accepted/_rejected message invalid in non-work-mode!" << std::endl;
              }

            }
          }
        }
      }

     private:
      virtual void sync_worker(std::size_t /*worker_id*/) {};

      virtual std::size_t num_tasks_from_cpus() {
        return 1;
      };


     protected:
      void gather_worker_results() {
        if (!retrieved) {
          ipqm->retrieve();
          for (auto const &item : ipqm->get_results()) {
            if (item.first.first == job_id) {
              ipqm_results[item.first.second] = item.second;
            }
          }
        }
      }

      std::size_t _NumCPU;
      InterProcessQueueAndMessenger *ipqm = nullptr;
      std::size_t job_id;
      std::map<Task, double> ipqm_results;
      bool retrieved = false;

      static bool work_mode;
    };

    // initialize static member
    template <typename Base> bool Vector<Base>::work_mode = true;
  }
}


using RooFit::MultiProcess::JobTask;

class xSquaredPlusBVectorParallel : public RooFit::MultiProcess::Vector<xSquaredPlusBVectorSerial> {
  using BASE = RooFit::MultiProcess::Vector<xSquaredPlusBVectorSerial>;
 public:
  xSquaredPlusBVectorParallel(std::size_t NumCPU, double b_init, std::vector<double> x_init) :
      BASE(NumCPU, b_init,
           x_init) // NumCPU stands for everything that defines the parallelization behaviour (number of cpu, strategy, affinity etc)
  {}

  void evaluate() override {
    // choose parallel strategy from multiprocess vector

    // sync remote first: local b -> workers
    sync();

    // master fills queue with tasks
    retrieved = false;
    for (std::size_t ix = 0; ix < x.size(); ++ix) {
      JobTask job_task(job_id, ix);
      ipqm->to_queue(job_task);
    }

    // wait for task results back from worker to master
    gather_worker_results();
    // put task results in desired container
    for (std::size_t ix = 0; ix < x.size(); ++ix) {
      result[ix] = ipqm_results[ix];
    }
  }


  void set_b_workers(double /*b*/) {}


  void sync() {
    // implementation defines sync, in this case update b only
  }


 private:
  void evaluate_task(std::size_t task_index) override {
    assert(ipqm->is_worker());
    result[task_index] = std::pow(x[task_index], 2) + _b.getVal();
  } // if serial implementation doesn't define evaluate_task -> implement here

  double get_task_result(std::size_t task) override {
    assert(ipqm->is_worker());
    return result[task];
  }

  std::size_t num_tasks_from_cpus() override {
    return _NumCPU;
  }

};


TEST(MultiProcess_Vector, xSquaredPlusB) {
  // Simple test case: calculate x^2 + b, where x is a vector. This case does
  // both a simple calculation (squaring the input vector x) and represents
  // handling of state updates in b.
  std::vector<double> x{0, 1, 2, 3};
  double b_initial = 3.;

  xSquaredPlusBVectorSerial x_sq_plus_b(b_initial, x);

  auto y = x_sq_plus_b.get_result();
  std::vector<double> y_expected{3, 4, 7, 12};

  EXPECT_EQ(y[0], y_expected[0]);
  EXPECT_EQ(y[1], y_expected[1]);
  EXPECT_EQ(y[2], y_expected[2]);
  EXPECT_EQ(y[3], y_expected[3]);

  std::size_t NumCPU = 1;
  xSquaredPlusBVectorParallel x_sq_plus_b_parallel(NumCPU, b_initial, x);
  xSquaredPlusBVectorParallel x_sq_plus_b_parallel2(NumCPU, b_initial, x);
  x_sq_plus_b_parallel.initialize_parallel_work_system();

  auto y_parallel = x_sq_plus_b_parallel.get_result();

  EXPECT_EQ(y_parallel[0], y_expected[0]);
  EXPECT_EQ(y_parallel[1], y_expected[1]);
  EXPECT_EQ(y_parallel[2], y_expected[2]);
  EXPECT_EQ(y_parallel[3], y_expected[3]);
}
