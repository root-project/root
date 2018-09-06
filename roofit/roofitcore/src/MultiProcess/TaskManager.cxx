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

#include <memory>  // make_shared
#include <stdexcept>  // logic_error
#include <sstream>

#include <ROOT/RMakeUnique.hxx>  // make_unique in C++11
#include <MultiProcess/TaskManager.h>
#include <MultiProcess/Job.h>
#include <MultiProcess/messages.h>

namespace RooFit {
  namespace MultiProcess {

    // static function
    TaskManager* TaskManager::instance(std::size_t N_workers) {
      if (!_instance) {
        assert(N_workers != 0);
        _instance = std::make_unique<TaskManager>(N_workers);
      } else {
        // some sanity checks
        if(_instance->is_master() && N_workers != _instance->worker_pipes.size()) {
          std::cerr << "On PID " << getpid() << ": N_workers != tmp->worker_pipes.size())! N_workers = " << N_workers << ", tmp->worker_pipes.size() = " << _instance->worker_pipes.size() << std::endl;
          throw std::logic_error("");
        } else if (_instance->is_worker()) {
          if (_instance->get_worker_id() + 1 != _instance->worker_pipes.size()) {
            std::cerr << "On PID " << getpid() << ": tmp->get_worker_id() + 1 != tmp->worker_pipes.size())! tmp->get_worker_id() = " << _instance->get_worker_id() << ", tmp->worker_pipes.size() = " << _instance->worker_pipes.size() << std::endl;
            throw std::logic_error("");
          }

          // use of shared_ptrs in combination with BidirMMapPipe's self-cleanup functionality makes the following test a bit harder to do, so we leave it as an exercise for the reader:
//          if (1 != std::accumulate(tmp->worker_pipes.begin(), tmp->worker_pipes.end(), 0,
//                                   [](int a, std::shared_ptr<BidirMMapPipe>& b) {
//                                     return a + (b->closed() ? 1 : 0);
//                                   })) {
//            std::cerr << "On PID " << getpid() << ": worker has multiple open worker pipes, should only be one!" << std::endl;
//            throw std::logic_error("");
//          }

        }
      }
      return _instance.get();
    }

    // static function
    TaskManager* TaskManager::instance() {
      if (!_instance) {
        throw std::runtime_error("in TaskManager::instance(): no instance was created yet! Call TaskManager::instance(std::size_t N_workers) first.");
      }
      return _instance.get();
    }

    // static function
    bool TaskManager::is_instantiated() {
      return static_cast<bool>(_instance);
    }


    void TaskManager::identify_processes() {
      // identify yourselves (for debugging)
      if (instance()->is_worker()) {
        std::cout << "I'm a worker, PID " << getpid() << std::endl;
      } else if (instance()->is_master()) {
        std::cout << "I'm master, PID " << getpid() << std::endl;
      } else if (instance()->is_queue()) {
        std::cout << "I'm queue, PID " << getpid() << std::endl;
      }
    }

    // constructor
    // Don't construct IPQM objects manually, use the static instance if
    // you need to run multiple jobs.
    TaskManager::TaskManager(std::size_t N_workers) {
      // This class defines three types of processes:
      // 1. master: the initial main process. It defines and enqueues tasks
      //    and processes results.
      // 2. workers: a pool of processes that will try to take tasks from the
      //    queue. These are first forked from master.
      // 3. queue: communication between the other types (necessarily) goes
      //    through this process. This process runs the queue_loop and
      //    maintains the queue of tasks. It is forked last and initialized
      //    with third BidirMMapPipe parameter false, which makes it the
      //    process that manages all pipes, though the pool of pages remains
      //    on the master process.
      // The reason for using this layout is that we use BidirMMapPipe for
      // forking and communicating between processes, and BidirMMapPipe only
      // supports forking from one process, not from an already forked
      // process (if forked using BidirMMapPipe). The latter layout would
      // allow us to fork first the queue from the main process and then fork
      // the workers from the queue, which may feel more natural.
      //
      // The master/queue processes pipe is created in initialization. Here
      // we manually create workers from the queue process.


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

      worker_pipes.reserve(N_workers);
      for (std::size_t ix = 0; ix < N_workers; ++ix) {
        // set worker_id before each fork so that fork will sync it to the worker
        worker_id = ix;
        worker_pipes.push_back(std::make_shared<BidirMMapPipe>(useExceptions, useSocketpair, keepLocal_WORKER));
        if(worker_pipes.back()->isChild()) {
          this_worker_pipe = worker_pipes.back();
          break;
        }
      }

      // then do the queue and master initialization, but each worker should
      // exit the constructor from here on
      if (worker_pipes.back()->isParent()) {
        queue_pipe = std::make_shared<BidirMMapPipe>(useExceptions, useSocketpair, keepLocal_QUEUE);

        if (queue_pipe->isParent()) {
          _is_master = true;
        } else if (queue_pipe->isChild()) {
          _is_queue = true;
        } else {
          // should never get here...
          throw std::runtime_error("Something went wrong while creating TaskManager!");
        }
      }
    }


    TaskManager::~TaskManager() {
      // The TM instance gets created by some Job. Once all Jobs are gone, the
      // TM will get destroyed. In this case, the job_objects map should have
      // been emptied. This check makes sure:
      assert(TaskManager::job_objects.size() == 0);
      terminate();
    }


    // static function
    // returns job_id for added job_object
    std::size_t TaskManager::add_job_object(Job *job_object) {
      if (TaskManager::is_instantiated()) {
        if (_instance->is_activated()) {
          std::stringstream ss;
          ss << "Cannot add Job to activated TaskManager instantiation (forking has already taken place)! Instance object at raw ptr " << _instance.get();
          throw std::logic_error("Cannot add Job to activated TaskManager instantiation (forking has already taken place)! Call terminate() on the instance before adding new Jobs.");
        }
      }
      std::size_t job_id = job_counter++;
      job_objects[job_id] = job_object;
      return job_id;
    }

    // static function
    Job* TaskManager::get_job_object(std::size_t job_object_id) {
      return job_objects[job_object_id];
    }

    // static function
    bool TaskManager::remove_job_object(std::size_t job_object_id) {
      bool removed_succesfully = job_objects.erase(job_object_id) == 1;
      if (job_objects.size() == 0) {
        _instance.reset(nullptr);
      }
      return removed_succesfully;
    }


    void TaskManager::terminate() noexcept {
      try {
        if (_is_master && queue_pipe->good()) {
          *queue_pipe << M2Q::terminate << BidirMMapPipe::flush;
          int retval = queue_pipe->close();
          if (0 != retval) {
            std::cerr << "error terminating queue_pipe" << "; child return value is " << retval << std::endl;
          }
        }
        queue_activated = false;
      } catch (const BidirMMapPipe::Exception& e) {
        std::cerr << "WARNING: in TaskManager::terminate, something in BidirMMapPipe threw an exception! Message:\n\t" << e.what() << std::endl;
      } catch (...) {
        std::cerr << "WARNING: something unknown in TaskManager::terminate (probably something in BidirMMapPipe) threw an exception!" << std::endl;
      }
    }

    void TaskManager::terminate_workers() {
      if (_is_queue) {
        for (std::shared_ptr<BidirMMapPipe> &worker_pipe : worker_pipes) {
          *worker_pipe << Q2W::terminate << BidirMMapPipe::flush;
        }
      }
    }


    // start message loops on child processes and quit processes afterwards
    void TaskManager::activate() {
      // should be called soon after creation of this object, because everything in
      // between construction and activate gets executed both on the master process
      // and on the slaves
      queue_activated = true; // set on all processes, master, queue and slaves

      if (_is_queue) {
        queue_loop();
        terminate_workers();
        std::_Exit(0);
      }
    }


    bool TaskManager::is_activated() {
      return queue_activated;
    }


    BidirMMapPipe::PollVector TaskManager::get_poll_vector() {
      BidirMMapPipe::PollVector poll_vector;
      poll_vector.reserve(1 + worker_pipes.size());
      poll_vector.emplace_back(queue_pipe.get(), BidirMMapPipe::Readable);
      for (std::shared_ptr<BidirMMapPipe>& pipe : worker_pipes) {
        poll_vector.emplace_back(pipe.get(), BidirMMapPipe::Readable);
      }
      return poll_vector;
    }


    bool TaskManager::process_queue_pipe_message(M2Q message) {
      bool carry_on = true;

      switch (message) {
        case M2Q::terminate: {
          carry_on = false;
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
          if (queue.empty() && N_tasks_completed == N_tasks) {
            *queue_pipe << Q2M::retrieve_accepted;  // handshake message (master will now start reading from the pipe)
            *queue_pipe << job_objects.size();
            for (auto job_tuple : job_objects) {
              *queue_pipe << job_tuple.first;  // job id
              job_tuple.second->send_back_results_from_queue_to_master();  // N_job_tasks, task_ids and results
              job_tuple.second->clear_results();
            }
            // reset number of received and completed tasks
            N_tasks = 0;
            N_tasks_completed = 0;
            *queue_pipe << BidirMMapPipe::flush;
          } else {
            *queue_pipe << Q2M::retrieve_rejected << BidirMMapPipe::flush;  // handshake message: tasks not done yet, try again
          }
        }
          break;

        case M2Q::update_real: {
          std::size_t job_id, ix;
          double val;
          bool is_constant;
          *queue_pipe >> job_id >> ix >> val >> is_constant;
          for (std::shared_ptr<BidirMMapPipe> &worker_pipe : worker_pipes) {
            *worker_pipe << Q2W::update_real << job_id << ix << val << is_constant << BidirMMapPipe::flush;
          }
        }
          break;

        case M2Q::switch_work_mode: {
          for (std::shared_ptr<BidirMMapPipe> &worker_pipe : worker_pipes) {
            *worker_pipe << Q2W::switch_work_mode << BidirMMapPipe::flush;
          }
        }
          break;

        case M2Q::call_double_const_method: {
          std::size_t job_id, worker_id_call;
          std::string key;
          *queue_pipe >> job_id >> worker_id_call >> key;
          *worker_pipes[worker_id_call] << Q2W::call_double_const_method << job_id << key << BidirMMapPipe::flush;
          double result;
          *worker_pipes[worker_id_call] >> result;
          *queue_pipe << result << BidirMMapPipe::flush;
        }
          break;
      }

      return carry_on;
    }


    void TaskManager::retrieve() {
      if (_is_master) {
        bool carry_on = true;
        while (carry_on) {
          *queue_pipe << M2Q::retrieve << BidirMMapPipe::flush;
          Q2M handshake;
          *queue_pipe >> handshake;
          if (handshake == Q2M::retrieve_accepted) {
            carry_on = false;
            std::size_t N_jobs;
            *queue_pipe >> N_jobs;
            for (std::size_t job_ix = 0; job_ix < N_jobs; ++job_ix) {
              std::size_t job_object_id;
              *queue_pipe >> job_object_id;
              TaskManager::get_job_object(job_object_id)->receive_results_on_master();
            }
          }
        }
      }
    }


    double TaskManager::call_double_const_method(std::string method_key, std::size_t job_id, std::size_t worker_id_call) {
      *queue_pipe << M2Q::call_double_const_method << job_id << worker_id_call << method_key << BidirMMapPipe::flush;
      double result;
      *queue_pipe >> result;
      return result;
    }

    // -- WORKER - QUEUE COMMUNICATION --

    void TaskManager::send_from_worker_to_queue() {
      *this_worker_pipe << BidirMMapPipe::flush;
    }

    void TaskManager::send_from_queue_to_worker(std::size_t this_worker_id) {
      *worker_pipes[this_worker_id] << BidirMMapPipe::flush;
    }

    // -- QUEUE - MASTER COMMUNICATION --

    void TaskManager::send_from_queue_to_master() {
      *queue_pipe << BidirMMapPipe::flush;
    }

    void TaskManager::send_from_master_to_queue() {
      send_from_queue_to_master();
    }


    void TaskManager::process_worker_pipe_message(BidirMMapPipe &pipe, std::size_t this_worker_id, W2Q message) {
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
          // receive back task result
          pipe >> job_object_id >> task;
          TaskManager::get_job_object(job_object_id)->receive_task_result_on_queue(task, this_worker_id);
          pipe << Q2W::result_received << BidirMMapPipe::flush;
          N_tasks_completed++;
          break;
        }
      }
    }


    void TaskManager::queue_loop() {
      if (_is_queue) {
        bool carry_on = true;
        auto poll_vector = get_poll_vector();

        while (carry_on) {
          // poll: wait until status change (-1: infinite timeout)
          int n_changed_pipes = BidirMMapPipe::poll(poll_vector, -1);
          // then process messages from changed pipes; loop while there are
          // still messages remaining after processing one (so that all
          // messages since changed pipe will get read).
          // TODO: Should we do this outside of the for loop, to make sure that both the master and the worker pipes are read from successively, instead of possibly getting stuck on one pipe only?
          // scan for pipes with changed status:
          std::size_t pipe_ix = 0;
          for (auto it = poll_vector.begin(); n_changed_pipes > 0 && poll_vector.end() != it; ++it, ++pipe_ix) {
            if (!it->revents) {
              // unchanged, next one
              continue;
            }
//              --n_changed_pipes; // maybe we can stop early...
            // read from pipes which are readable
            if (it->revents & BidirMMapPipe::Readable) {
              // message comes from the master/queue pipe (first element):
              if (it == poll_vector.begin()) {
                M2Q message;
                *queue_pipe >> message;
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
                process_worker_pipe_message(pipe, pipe_ix - 1, message);
              }
            }
          }
        }
      }
    }


    // Have a worker ask for a task-message from the queue
    bool TaskManager::from_queue(JobTask &job_task) {
      if (queue.empty()) {
        return false;
      } else {
        job_task = queue.front();
        queue.pop();
        return true;
      }
    }


    // Enqueue a task
    void TaskManager::to_queue(JobTask job_task) {
      if (is_master()) {
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


    bool TaskManager::is_master() {
      return _is_master;
    }

    bool TaskManager::is_queue() {
      return _is_queue;
    }

    bool TaskManager::is_worker() {
      return !(_is_master || _is_queue);
    }

    void TaskManager::set_work_mode(bool flag) {
      if (is_master() && flag != work_mode) {
        work_mode = flag;
        *queue_pipe << M2Q::switch_work_mode << BidirMMapPipe::flush;
      }
    }

    std::size_t TaskManager::get_worker_id() {
      return worker_id;
    }


    // initialize static members
    std::map<std::size_t, Job *> TaskManager::job_objects;
    std::size_t TaskManager::job_counter = 0;
    std::unique_ptr<TaskManager> TaskManager::_instance {nullptr};

  } // namespace MultiProcess
} // namespace RooFit