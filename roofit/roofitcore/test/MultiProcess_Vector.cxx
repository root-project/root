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
#include <memory>  // make_shared
#include <algorithm>  // all_of

#include <RooRealVar.h>
#include <../src/BidirMMapPipe.h>

// for NLL tests
#include <TRandom.h>
#include <RooWorkspace.h>
#include <RooAbsPdf.h>
#include <RooDataSet.h>
#include <RooNLLVar.h>

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
  namespace util {
    template <typename C>
    bool all_true(C& container) {
      static_assert(std::is_same<typename C::value_type, bool>::value, "Container must have value_type bool");
      return std::all_of(container.begin(), container.end(), [](bool i){return i;});
    }
  }

  namespace MultiProcess {

    // Messages from master to queue
    enum class M2Q : int {
      terminate = 100,
      enqueue = 10,
      retrieve = 11//,
//      sync_job_results_to_master = 198,
//      return_control_to_master = 199
    };

    // Messages from queue to master
    enum class Q2M : int {
      terminate = 200,
      retrieve_rejected = 20,
      retrieve_accepted = 21
    };

    // Messages from worker to queue
    enum class W2Q : int {
      terminate = 300,
      dequeue = 30,
      send_result = 31
    };

    // Messages from queue to worker
    enum class Q2W : int {
      terminate = 400,
      dequeue_rejected = 40,
      dequeue_accepted = 41,
      update_parameter = 42,
      switch_work_mode = 43
    };

    // for debugging
#define PROCESS_VAL(p) case(p): s = #p; break;

    std::ostream& operator<<(std::ostream& out, const M2Q value){
      const char* s = 0;
      switch(value){
        PROCESS_VAL(M2Q::terminate);
        PROCESS_VAL(M2Q::enqueue);
        PROCESS_VAL(M2Q::retrieve);
      }
      return out << s;
    }

    std::ostream& operator<<(std::ostream& out, const Q2M value){
      const char* s = 0;
      switch(value){
        PROCESS_VAL(Q2M::terminate);
        PROCESS_VAL(Q2M::retrieve_rejected);
        PROCESS_VAL(Q2M::retrieve_accepted);
      }
      return out << s;
    }

    std::ostream& operator<<(std::ostream& out, const W2Q value){
      const char* s = 0;
      switch(value){
        PROCESS_VAL(W2Q::terminate);
        PROCESS_VAL(W2Q::dequeue);
        PROCESS_VAL(W2Q::send_result);
      }
      return out << s;
    }

    std::ostream& operator<<(std::ostream& out, const Q2W value){
      const char* s = 0;
      switch(value){
        PROCESS_VAL(Q2W::terminate);
        PROCESS_VAL(Q2W::dequeue_rejected);
        PROCESS_VAL(Q2W::dequeue_accepted);
        PROCESS_VAL(Q2W::update_parameter);
        PROCESS_VAL(Q2W::switch_work_mode);
      }
      return out << s;
    }

#undef PROCESS_VAL

  }
}


// stream operators for message enum classes
namespace RooFit {

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
//      virtual void sync_job_results_to_master() = 0;
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


    class InterProcessQueueAndMessenger {
     public:
      static std::shared_ptr<InterProcessQueueAndMessenger> instance(std::size_t NumCPU) {
        if (!_instance) {
          assert(NumCPU != 0);
          _instance = std::make_shared<InterProcessQueueAndMessenger>(NumCPU);
        } else {
          assert(NumCPU == _instance->worker_pipes.size());
        }
        return _instance;
      }

      static std::shared_ptr<InterProcessQueueAndMessenger> instance() {
        assert(_instance);
        return _instance;
      }

      // constructor
      // Don't construct IPQM objects manually, use the static instance if
      // you need to run multiple jobs.
      explicit InterProcessQueueAndMessenger(std::size_t NumCPU) {
        // This class defines three types of processes:
        // 1. queue: this is the initial main process; communication between
        //    the other types (necessarily) goes through this process. This
        //    process runs the queue_loop and maintains the queue of tasks.
        // 2. director: takes over the role of "main" process during parallel
        //    processing. It defines and enqueues tasks and processes results.
        // 3. workers: a pool of processes that will try to take tasks from the
        //    queue.
        // The reason for using this layout is that we use BidirMMapPipe for
        // forking and communicating between processes, and BidirMMapPipe only
        // supports forking from one process, not from an already forked
        // process (if forked using BidirMMapPipe). The latter layout would
        // allow us to fork first the queue from the main process and then fork
        // the workers from the queue, which may feel more natural.
        //
        // The director/queue processes pipe is created in initialization. Here
        // we manually create workers from the queue process.

        // NEW IMPLEMENTATION:
        // 1. director == master
        // 2. workers are first forked from master
        // 3. queue is forked last and initialized with third parameter false,
        //    which makes it the process that manages all pipes and the pool of
        //    pages.
        // Again this works around the BidirMMapPipe restrictions, but due to
        // this new bipe option, a more logical control flow layout can be
        // maintained.

        // BidirMMapPipe parameters
        bool useExceptions = true;
        bool useSocketpair = false;
        bool keepLocal_WORKER = true;
        bool keepLocal_QUEUE = false;

        // First initialize the workers.
        // Reserve is necessary! BidirMMapPipe is not allowed to be copied,
        // but when capacity is not enough when using emplace_back, the
        // vector must be resized, which means existing elements must be
        // copied to the new memory locations.

        worker_pipes.reserve(NumCPU);
        for (std::size_t ix = 0; ix < NumCPU; ++ix) {
          // set worker_id before each fork so that fork will sync it to the worker
          worker_id = ix;
          worker_pipes.push_back(std::make_shared<BidirMMapPipe>(useExceptions, useSocketpair, keepLocal_WORKER));
        }

        // then do the queue and director initialization, but each worker should
        // exit the constructor from here on
        if (worker_pipes.back()->isParent()) {
          queue_pipe = std::make_shared<BidirMMapPipe>(useExceptions, useSocketpair, keepLocal_QUEUE);

          if (queue_pipe->isParent()) {
            _is_director = true;
          } else if (queue_pipe->isChild()) {
            _is_queue = true;
          } else {
            // should never get here...
            throw std::runtime_error("Something went wrong while creating InterProcessQueueAndMessenger!");
          }
        }
      }


      ~InterProcessQueueAndMessenger() {
        terminate();
      }

      // returns job_id for added job_object
      static std::size_t add_job_object(Job *job_object) {
        job_objects.push_back(job_object);
        job_returned.push_back(false);
        return job_objects.size() - 1;
      }

      static Job* get_job_object(std::size_t job_object_id) {
        return job_objects[job_object_id];
      }

      // protocol for terminating processes: send terminate message and wait for child
      // to die, which we check using the pipe's eof method.
      template <typename Send>
      static void terminate_pipe(BidirMMapPipe &pipe, std::string error_message) {
        pipe << Send::terminate << BidirMMapPipe::flush;
        bool wait_for_eof = true;
        unsigned times_waited_for_eof = 0;
        while (wait_for_eof && times_waited_for_eof < 10) {
          if (!pipe.eof()) {
            ++times_waited_for_eof;
            continue;
          } else {
            wait_for_eof = false;
            int retval = pipe.close();
            if (0 != retval) {
              std::cerr << error_message << "; child return value is " << retval << std::endl;
            }
          }
        }
      }


//      bool return_control_to_master(std::size_t job_object_id) {
//        assert(is_director());
//        *queue_pipe << M2Q::return_control_to_master << job_object_id << BidirMMapPipe::flush;
//        bool die;
//        *queue_pipe >> die;
//        return die;
//      }


      void terminate() {
        if (_is_director) {
          terminate_pipe<M2Q>(*queue_pipe, "In terminate: queue shutdown failed.");
        }
      }


      void terminate_director() {
        if (_is_queue) {
          if (0 != queue_pipe->close()) {
            std::cerr << "In terminate_director: director shutdown failed." << std::endl;
          }
        }
      }


      void terminate_workers() {
        if (_is_queue) {
          for (std::shared_ptr<BidirMMapPipe> &worker_pipe : worker_pipes) {
            std::stringstream ss;
            ss << "In terminate_workers: worker with PID " << worker_pipe->pidOtherEnd() << " shutdown failed.";
            terminate_pipe<Q2W>(*worker_pipe, ss.str());
          }
        }
      }


      // start message loops on child processes and quit processes afterwards
      void activate() {
        std::cout << "activate called from PID " << getpid() << std::endl;
        // should be called soon after creation of this object, because everything in
        // between construction and activate gets executed both on the master process
        // and on the slaves
        if (_is_director) {
          // on master we only have to set the activated flag so that we can start
          // queuing tasks
          queue_activated = true;
          // it's not important on the other processes
        } else if (_is_queue) {
          std::cout << "activate on PID " << getpid() << " starting queue_loop" << std::endl;
          queue_loop();
          // the queue_loop can end when all jobs sent return_control_to_master,
          // in which case the director already terminated itself
          std::cout << "activate on PID " << getpid() << " ended queue_loop, now terminating workers" << std::endl;
          terminate_workers();
        } else { // is worker
          queue_activated = true;
        }
      }


      BidirMMapPipe::PollVector get_poll_vector() {
        BidirMMapPipe::PollVector poll_vector;
        poll_vector.reserve(1 + worker_pipes.size());
        poll_vector.emplace_back(queue_pipe.get(), BidirMMapPipe::Readable);
        for (std::shared_ptr<BidirMMapPipe>& pipe : worker_pipes) {
          poll_vector.emplace_back(pipe.get(), BidirMMapPipe::Readable);
        }
        return poll_vector;
      }


      bool process_queue_pipe_message(M2Q message) {
        bool carry_on = true;

        switch (message) {
          case M2Q::terminate: {
            // NOTE: if/when implementing this, make sure that in activate() the
            // proper actions are taken to also terminate other processes.
            // Currently, activate() assumes there's only one way for the
            // queue_loop to end, which is through return_control_to_master, which
            // means the director has shut down itself already.
          }
          break;

          case M2Q::enqueue: {
            // enqueue task
            Task task;
            std::size_t job_object_id;
            *queue_pipe >> job_object_id >> task;
            JobTask job_task(job_object_id, task);
            to_queue(job_task);
            N_tasks++;
          }
          break;

          case M2Q::retrieve: {
            // retrieve task results after queue is empty and all
            // tasks have been completed
            if (queue.empty() && results.size() == N_tasks) {
              *queue_pipe << Q2M::retrieve_accepted;  // handshake message (master will now start reading from the pipe)
              *queue_pipe << N_tasks;
              for (auto const &item : results) {
                *queue_pipe << item.first.first << item.first.second << item.second;
              }
              // empty results cache
              results.clear();
              // reset number of received tasks
              N_tasks = 0;
              *queue_pipe << BidirMMapPipe::flush;
            } else {
              *queue_pipe << Q2M::retrieve_rejected << BidirMMapPipe::flush;  // handshake message: tasks not done yet, try again
            }
          }
          break;

//          case M2Q::sync_job_results_to_master: {
//            // this is necessary because master is the queue process due to a
//            // limitation in forking from a fork
//            std::size_t job_object_id;
//            *queue_pipe >> job_object_id;
//            get_job_object(job_object_id)->sync_job_results_to_master();
//          }
//          break;

//          case M2Q::return_control_to_master: {
//            // this is necessary because master is the queue process due to a
//            // limitation in forking from a fork
//            std::size_t job_object_id;
//            *queue_pipe >> job_object_id;
//            job_returned[job_object_id] = true;
//            if (util::all_true(job_returned)) {
//              carry_on = false;
//              *queue_pipe << true;  // let the director die
//            } else {
//              *queue_pipe << false; // let the director live
//            }
//            *queue_pipe << BidirMMapPipe::flush;
//          }
//          break;
        }

        return carry_on;
      }


      void retrieve() {
        if (_is_director) {
          std::cout << "retrieving..." << std::endl;
          bool carry_on = true;
          while (carry_on) {
            std::cout << "retrieve sent message " << M2Q::retrieve << std::endl;
            *queue_pipe << M2Q::retrieve << BidirMMapPipe::flush;
            Q2M handshake;
            *queue_pipe >> handshake;
            std::cout << "retrieve got handshake: " << handshake << std::endl;
            if (handshake == Q2M::retrieve_accepted) {
              carry_on = false;
              *queue_pipe >> N_tasks;
              for (std::size_t ix = 0; ix < N_tasks; ++ix) {
                Task task;
                std::size_t job_object_id;
                double result;
                *queue_pipe >> job_object_id >> task >> result;
                JobTask job_task(job_object_id, task);
                results[job_task] = result;
              }
            }
          }
        }
        std::cout << "retrieved." << std::endl;
      }


      void process_worker_pipe_message(BidirMMapPipe &pipe, W2Q message) {
        Task task;
        std::size_t job_object_id;
        switch (message) {
          case W2Q::dequeue: {
            // dequeue task
            JobTask job_task;
            if (from_queue(job_task)) {
              pipe << Q2W::dequeue_accepted << job_task.first << job_task.second;
            } else {
              pipe << Q2W::dequeue_rejected;
            }
            pipe << BidirMMapPipe::flush;
            break;
          }

          case W2Q::send_result: {
            // receive back task result and store it for later
            // sending back to master
            // TODO: add RooAbsCategory handling!
            std::cout << "queue receiving result from worker" << std::endl;
            double result;
            pipe >> job_object_id >> task >> result;
            std::cout << "queue received result from worker: " << result << ", from job/task " << job_object_id << "/" << task << std::endl;
            JobTask job_task(job_object_id, task);
            results[job_task] = result;
            break;
          }

          case W2Q::terminate: {
            throw std::runtime_error("In queue_loop: received terminate signal from worker, but this signal should only be sent as handshake after queue sends terminate signal first!");
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
            // then process messages from changed pipes; loop while there are
            // still messages remaining after processing one (so that all
            // messages since changed pipe will get read).
            // TODO: Should we do this outside of the for loop, to make sure that both the director and the worker pipes are read from successively, instead of possibly getting stuck on one pipe only?
            // scan for pipes with changed status:
            for (auto it = poll_vector.begin(); n_changed_pipes > 0 && poll_vector.end() != it; ++it) {
              if (!it->revents) {
                // unchanged, next one
                continue;
              }
//              --n_changed_pipes; // maybe we can stop early...
              // read from pipes which are readable
              do { // while there are bytes to read
                if (it->revents & BidirMMapPipe::Readable) {
//                  std::cout << "hi" << std::endl;
                  // message comes from the director/queue pipe (first element):
                  if (it == poll_vector.begin()) {
                    M2Q message;
                    *queue_pipe >> message;
//                    std::cout << "queue got message " << message << std::endl;
                    carry_on = process_queue_pipe_message(message);
                    // on terminate, also stop for-loop, no need to check other
                    // pipes anymore:
                    if (!carry_on) {
                      n_changed_pipes = 0;
                    }
                  } else { // from a worker pipe
                    W2Q message;
                    BidirMMapPipe &pipe = *(it->pipe);
                    pipe >> message;
//                    std::cout << "queue got message " << message << std::endl;
                    process_worker_pipe_message(pipe, message);
                  }
                }
              } while (it->pipe->bytesReadableNonBlocking() > 0);
            }
          }
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
        if (is_director()) {
          if (!queue_activated) {
            activate();
          }
          *queue_pipe << M2Q::enqueue << job_task.first << job_task.second << BidirMMapPipe::flush;

        } else if (is_queue()) {
          queue.push(job_task);
        } else {
          throw std::logic_error("calling Communicator::to_master_queue from slave process");
        }
      }


      bool is_director() {
        return _is_director;
      }

      bool is_queue() {
        return _is_queue;
      }

      bool is_worker() {
        return !(_is_director || _is_queue);
      }


      std::shared_ptr<BidirMMapPipe>& get_worker_pipe() {
        assert(is_worker());
        return worker_pipes[worker_id];
      }

      std::size_t get_worker_id() {
        return worker_id;
      }


      std::shared_ptr<BidirMMapPipe>& get_queue_pipe() {
        assert(is_director() || is_queue());
        return queue_pipe;
      }


      std::map<JobTask, double>& get_results() {
        return results;
      }

     public:



     private:
      std::vector< std::shared_ptr<BidirMMapPipe> > worker_pipes;
      std::shared_ptr<BidirMMapPipe> queue_pipe;
      std::size_t worker_id;
      bool _is_director = false;
      bool _is_queue = false;
      std::queue<JobTask> queue;
      std::size_t N_tasks = 0;  // total number of received tasks
      // TODO: add RooAbsCategory handling to results!
      std::map<JobTask, double> results;
      bool queue_activated = false;

      static std::vector<Job *> job_objects;
      static std::vector<bool> job_returned;
      static std::shared_ptr<InterProcessQueueAndMessenger> _instance;
    };

    // initialize static members
    std::vector<Job *> InterProcessQueueAndMessenger::job_objects;
    std::vector<bool> InterProcessQueueAndMessenger::job_returned;
    std::shared_ptr<InterProcessQueueAndMessenger> InterProcessQueueAndMessenger::_instance {};

    // Vector defines an interface and communication machinery to build a
    // parallelized subclass of an existing non-concurrent numerical class that
    // can be expressed as a vector of independent sub-calculations.
    //
    // TODO: update documentation for new InterProcessQueueAndMessenger!
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
          _NumCPU(NumCPU)
      {
        job_id = InterProcessQueueAndMessenger::add_job_object(this);
      }

      ~Vector() {
//        delete ipqm;
      }

      void initialize_parallel_work_system() {
        if (!ipqm) {
          ipqm = InterProcessQueueAndMessenger::instance(_NumCPU);
        }

        ipqm->activate();

        if (ipqm->is_worker()) {
          std::cout << "worker PID " << getpid() << " starting worker_loop" << std::endl;
          worker_loop();
          std::_Exit(0);
        }
      }

      static void worker_loop() {
        assert(InterProcessQueueAndMessenger::instance()->is_worker());
        bool carry_on = true;
        Task task;
        std::size_t job_object_id;
        Q2W message_q2w;
        JobTask job_task;
        BidirMMapPipe& pipe = *InterProcessQueueAndMessenger::instance()->get_worker_pipe();
        while (carry_on) {
          if (work_mode) {
            // try to dequeue a task
            pipe << W2Q::dequeue << BidirMMapPipe::flush;
//            std::cout << "worker " << InterProcessQueueAndMessenger::instance().get_worker_id() << " sent message " << W2Q::dequeue << " to queue" << std::endl;

            // receive handshake
            pipe >> message_q2w;
//            std::cout << "worker " << InterProcessQueueAndMessenger::instance().get_worker_id() << " got message " << message_q2w << std::endl;

            switch (message_q2w) {
              case Q2W::terminate: {
                carry_on = false;
                break;
              }

              case Q2W::dequeue_rejected: {
                break;
              }
              case Q2W::dequeue_accepted: {
                pipe >> job_object_id >> task;
                std::cout << "worker " << InterProcessQueueAndMessenger::instance()->get_worker_id()
                          << " evaluating jobtask " << job_object_id << "/" << task << std::endl;
                InterProcessQueueAndMessenger::get_job_object(job_object_id)->evaluate_task(task);

                // TODO: add RooAbsCategory handling!
                double result = InterProcessQueueAndMessenger::get_job_object(job_object_id)->get_task_result(task);
                std::cout << "worker " << InterProcessQueueAndMessenger::instance()->get_worker_id()
                          << " sending result " << result << " for jobtask " << job_object_id << "/" << task << std::endl;
                pipe << W2Q::send_result << job_object_id << task << result << BidirMMapPipe::flush;

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
            std::cout << "worker " << InterProcessQueueAndMessenger::instance()->get_worker_id() << " got message " << message_q2w << std::endl;

            switch (message_q2w) {
              case Q2W::terminate: {
                pipe << W2Q::terminate << BidirMMapPipe::flush;
                carry_on = false;
                break;
              }

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
      std::shared_ptr<InterProcessQueueAndMessenger> ipqm;
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
 public:
  xSquaredPlusBVectorParallel(std::size_t NumCPU, double b_init, std::vector<double> x_init) :
      RooFit::MultiProcess::Vector<xSquaredPlusBVectorSerial>(NumCPU, b_init,
           x_init) // NumCPU stands for everything that defines the parallelization behaviour (number of cpu, strategy, affinity etc)
  {}

//  void sync_job_results_to_master() override {
//    if (ipqm->is_director()) {
//      ipqm->get_queue_pipe() << RooFit::MultiProcess::M2Q::sync_job_results_to_master;
//      ipqm->get_queue_pipe() << job_id;  // job_id is read in queue_loop and used to dispatch back to this job on queue process
//      for (std::size_t ix = 0; ix < x.size(); ++ix) {
//        ipqm->get_queue_pipe() << result[ix];
//      }
//      ipqm->get_queue_pipe() << RooFit::BidirMMapPipe::flush;
//    } else if (ipqm->is_queue()) {
//      for (std::size_t ix = 0; ix < x.size(); ++ix) {
//        ipqm->get_queue_pipe() >> result[ix];
//      }
//    } else {
//      throw std::logic_error("sync_job_results_to_master called from process that is not director nor queue!");
//    }
//  }

  void evaluate() override {
    if (ipqm->is_director()) {
      // sync remote first: local b -> workers
      sync();

      // master fills queue with tasks
      retrieved = false;
      for (std::size_t ix = 0; ix < x.size(); ++ix) {
        JobTask job_task(job_id, ix);
        ipqm->to_queue(job_task);
      }

      // wait for task results back from workers to director
      gather_worker_results();
      // put task results in desired container
      for (std::size_t ix = 0; ix < x.size(); ++ix) {
        result[ix] = ipqm_results[ix];
      }

//      sync_job_results_to_master();
//      bool die = ipqm->return_control_to_master(job_id);
//      if (die) {
//        std::_Exit(0);
//      }
    }
  }


  void sync() {
    // implementation defines sync, in this case update b only
  }


 private:
  void evaluate_task(std::size_t task) override {
    assert(ipqm->is_worker());
    result[task] = std::pow(x[task], 2) + _b.getVal();
  }

  double get_task_result(std::size_t task) override {
    assert(ipqm->is_worker());
    return result[task];
  }

};


TEST(MultiProcessVector, getResultSINGLEJOB) {
  // Simple test case: calculate x^2 + b, where x is a vector. This case does
  // both a simple calculation (squaring the input vector x) and represents
  // handling of state updates in b.
  std::vector<double> x{0, 1, 2, 3};
  double b_initial = 3.;

  // start serial test

  xSquaredPlusBVectorSerial x_sq_plus_b(b_initial, x);

  auto y = x_sq_plus_b.get_result();
  std::vector<double> y_expected{3, 4, 7, 12};

  EXPECT_EQ(y[0], y_expected[0]);
  EXPECT_EQ(y[1], y_expected[1]);
  EXPECT_EQ(y[2], y_expected[2]);
  EXPECT_EQ(y[3], y_expected[3]);

  std::size_t NumCPU = 1;

  // start parallel test

  xSquaredPlusBVectorParallel x_sq_plus_b_parallel(NumCPU, b_initial, x);
  x_sq_plus_b_parallel.initialize_parallel_work_system();

  std::cout << "calling get_result on PID " << getpid() << std::endl;

  auto y_parallel = x_sq_plus_b_parallel.get_result();
  std::cout << "results (PID " << getpid() << "): "
            << y_parallel[0] << ", "
      << y_parallel[1] << ", "
      << y_parallel[2] << ", "
      << y_parallel[3] << std::endl;

  EXPECT_EQ(y_parallel[0], y_expected[0]);
  EXPECT_EQ(y_parallel[1], y_expected[1]);
  EXPECT_EQ(y_parallel[2], y_expected[2]);
  EXPECT_EQ(y_parallel[3], y_expected[3]);
}


TEST(MultiProcessVector, DISABLED_getResultMULTIJOB) {
  // Simple test case: calculate x^2 + b, where x is a vector. This case does
  // both a simple calculation (squaring the input vector x) and represents
  // handling of state updates in b.
  std::vector<double> x{0, 1, 2, 3};
  double b_initial = 3.;

  std::vector<double> y_expected{3, 4, 7, 12};

  std::size_t NumCPU = 1;

  // define jobs
  xSquaredPlusBVectorParallel x_sq_plus_b_parallel(NumCPU, b_initial, x);
  xSquaredPlusBVectorParallel x_sq_plus_b_parallel2(NumCPU, b_initial + 1, x);

  // do stuff
  x_sq_plus_b_parallel.initialize_parallel_work_system();

  auto y_parallel = x_sq_plus_b_parallel.get_result();
  auto y_parallel2 = x_sq_plus_b_parallel2.get_result();

  EXPECT_EQ(y_parallel[0], y_expected[0]);
  EXPECT_EQ(y_parallel[1], y_expected[1]);
  EXPECT_EQ(y_parallel[2], y_expected[2]);
  EXPECT_EQ(y_parallel[3], y_expected[3]);

  EXPECT_EQ(y_parallel2[0], y_expected[0] + 1);
  EXPECT_EQ(y_parallel2[1], y_expected[1] + 1);
  EXPECT_EQ(y_parallel2[2], y_expected[2] + 1);
  EXPECT_EQ(y_parallel2[3], y_expected[3] + 1);
}


enum class RooNLLVarTask {
  all_events,
  single_event,
  bulk_partition,
  interleave
};


class MPRooNLLVar : public RooFit::MultiProcess::Vector<RooNLLVar> {
 public:
  // use copy constructor for the RooNLLVar part
  MPRooNLLVar(std::size_t NumCPU, RooNLLVarTask task_mode, const RooNLLVar& nll) :
      RooFit::MultiProcess::Vector<RooNLLVar>(NumCPU, nll),
      mp_task_mode(task_mode)
  {
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
  }

//  void sync_job_results_to_master() override {
//    if (ipqm->is_director()) {
//      ipqm->get_queue_pipe() << RooFit::MultiProcess::M2Q::sync_job_results_to_master;
//      ipqm->get_queue_pipe() << job_id;  // job_id is read in queue_loop and used to dispatch back to this job on queue process
//      ipqm->get_queue_pipe() << result;
//      ipqm->get_queue_pipe() << RooFit::BidirMMapPipe::flush;
//    } else if (ipqm->is_queue()) {
//      ipqm->get_queue_pipe() >> result;
//    } else {
//      throw std::logic_error("sync_job_results_to_master called from process that is not director nor queue!");
//    }
//  }

  // the const is inherited from RooAbsTestStatistic::evaluate. We are not
  // actually const though, so we use a horrible hack.
  Double_t evaluate() const override {
    return const_cast<MPRooNLLVar*>(this)->evaluate_non_const();
  }

  Double_t evaluate_non_const() {
    if (ipqm->is_director()) {
      // sync remote first: local b -> workers
      sync();

      // master fills queue with tasks
      retrieved = false;
      for (std::size_t ix = 0; ix < N_tasks; ++ix) {
        JobTask job_task(job_id, ix);
        ipqm->to_queue(job_task);
      }

      // wait for task results back from workers to director
      gather_worker_results();
      // sum task results
      for (std::size_t ix = 0; ix < N_tasks; ++ix) {
        result += ipqm_results[ix];
      }

//      sync_job_results_to_master();
//      bool die = ipqm->return_control_to_master(job_id);
//      if (die) {
//        std::_Exit(0);
//      }
    }
    return result;
  }


  void sync() {
    // implementation defines sync, in this case update b only
  }


 private:
  void evaluate_task(std::size_t task) override {
    assert(ipqm->is_worker());
    std::size_t first, last, step;
    std::size_t N_events = static_cast<std::size_t>(_data->numEntries());
    switch (mp_task_mode) {
      case RooNLLVarTask::all_events: {
        first = task;
        last = N_events;
        step = 1;
        break;
      }
      case RooNLLVarTask::single_event: {
        first = task;
        last = task + 1;
        step = 1;
        break;
      }
      case RooNLLVarTask::bulk_partition: {
        first = N_events * task / N_tasks;
        last  = N_events * (task + 1) / N_tasks;
        step  = 1;
        break;
      }
      case RooNLLVarTask::interleave: {
        first = task;
        last = N_events;
        step = N_tasks;
        break;
      }
    }

    result = evaluatePartition(first, last, step);
  }

  double get_task_result(std::size_t /*task*/) override {
    assert(ipqm->is_worker());
    return result;
  }

  double result = 0;
  std::size_t N_tasks = 0;
  RooNLLVarTask mp_task_mode;
};


TEST(MultiProcessVectorNLL, getResultAllEvents) {
  // Real-life test: calculate a NLL using event-based parallelization. This
  // should replicate RooRealMPFE results.
  gRandom->SetSeed(1);
  RooWorkspace w;
  w.factory("Gaussian::g(x[-5,5],mu[0,-3,3],sigma[1])");
  auto x = w.var("x");
  RooAbsPdf *pdf = w.pdf("g");
  RooDataSet *data = pdf->generate(RooArgSet(*x), 10000);
  auto nll = pdf->createNLL(*data);

  std::size_t NumCPU = 1;
  RooNLLVarTask mp_task_mode = RooNLLVarTask::all_events;

  auto nominal_result = nll->getVal();

  MPRooNLLVar nll_mp(NumCPU, mp_task_mode, *dynamic_cast<RooNLLVar*>(nll));

  auto mp_result = nll_mp.getVal();

  EXPECT_EQ(nominal_result, mp_result);
}


