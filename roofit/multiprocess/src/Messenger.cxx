/*
 * Project: RooFit
 * Authors:
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl
 *   IP, Inti Pelupessy, Netherlands eScience Center, i.pelupessy@esciencecenter.nl
 *
 * Copyright (c) 2016-2019, Netherlands eScience Center
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#include "RooFit/MultiProcess/JobManager.h"
#include "RooFit/MultiProcess/Messenger.h"

namespace RooFit {
namespace MultiProcess {

void set_socket_immediate(ZmqLingeringSocketPtr<> &socket)
{
   int optval = 1;
   socket->setsockopt(ZMQ_IMMEDIATE, &optval, sizeof(optval));
}

Messenger::Messenger(const ProcessManager &process_manager)
{
   // create zmq connections (zmq context is automatically created in the ZeroMQSvc class and maintained as singleton)
   try {
      if (process_manager.is_master()) {
         mq_socket.reset(zmqSvc().socket_ptr(zmq::PAIR));
         set_socket_immediate(mq_socket);
         mq_socket->bind("ipc:///tmp/roofitMP_master_queue");
      } else if (process_manager.is_queue()) {
         // first the queue-worker sockets
         qw_sockets.resize(
            process_manager.N_workers()); // do resize instead of reserve so that the unique_ptrs are initialized
                                              // (to nullptr) so that we can do reset below, alternatively you can do
                                              // push/emplace_back with move or something
         for (std::size_t ix = 0; ix < process_manager.N_workers(); ++ix) {
            qw_sockets[ix].reset(zmqSvc().socket_ptr(zmq::PAIR));
            set_socket_immediate(qw_sockets[ix]);
            std::stringstream socket_name;
            socket_name << "ipc:///tmp/roofitMP_queue_worker_" << ix;
            qw_sockets[ix]->bind(socket_name.str());
         }
         // then the master-queue socket
         mq_socket.reset(zmqSvc().socket_ptr(zmq::PAIR));
         set_socket_immediate(mq_socket);
         mq_socket->connect("ipc:///tmp/roofitMP_master_queue");
      } else if (process_manager.is_worker()) {
         this_worker_qw_socket.reset(zmqSvc().socket_ptr(zmq::PAIR));
         set_socket_immediate(this_worker_qw_socket);
         std::stringstream socket_name;
         socket_name << "ipc:///tmp/roofitMP_queue_worker_" << process_manager.worker_id();
         this_worker_qw_socket->connect(socket_name.str());
      } else {
         // should never get here
         throw std::runtime_error("TaskManager::initialize_processes: I'm neither master, nor queue, nor a worker");
      }
   } catch (zmq::error_t &e) {
      std::cerr << e.what() << " -- errnum: " << e.num() << std::endl;
      throw;
   };
}

Messenger::~Messenger() {
   close_master_queue_connection();
   // the destructor is only used on the master process, so worker-queue
   // connections needn't be closed here; see documentation of JobManager
   // destructor
}


void Messenger::close_master_queue_connection() noexcept {
   try {
      if (JobManager::instance()->process_manager().is_master()
          || JobManager::instance()->process_manager().is_queue()) {
         mq_socket.reset(nullptr);
         zmqSvc().close_context();
      }
   } catch (const std::exception& e) {
      std::cerr << "WARNING: something in Messenger::terminate (on "
                << (JobManager::instance()->process_manager().is_master() ? "master" : "queue")
                << ") threw an exception! Original exception message:\n" << e.what() << std::endl;
   }
}


void Messenger::close_queue_worker_connections() {
   if (JobManager::instance()->process_manager().is_worker()) {
      this_worker_qw_socket.reset(nullptr);
      zmqSvc().close_context();
   } else if (JobManager::instance()->process_manager().is_queue()) {
      for (std::size_t worker_ix = 0ul; worker_ix < JobManager::instance()->process_manager().N_workers(); ++worker_ix) {
         qw_sockets[worker_ix].reset(nullptr);
      }
   }
}


std::pair<ZeroMQPoller, std::size_t> Messenger::create_poller() {
   ZeroMQPoller poller;
   std::size_t mq_index = poller.register_socket(*mq_socket, zmq::POLLIN);
   for (auto &s : qw_sockets) {
      poller.register_socket(*s, zmq::POLLIN);
   }
   return {std::move(poller), mq_index};
}


bool Messenger::is_initialized() const
{
   if (JobManager::instance()->process_manager().is_worker()) {
      return static_cast<bool>(this_worker_qw_socket);
   } else if (JobManager::instance()->process_manager().is_queue()) {
      return static_cast<bool>(qw_sockets[0]) && static_cast<bool>(mq_socket);
   } else if (JobManager::instance()->process_manager().is_master()) {
      return static_cast<bool>(mq_socket);
   } else {
      return false;
   }
}

// -- WORKER - QUEUE COMMUNICATION --

void Messenger::send_from_worker_to_queue() {}

void Messenger::send_from_queue_to_worker(std::size_t /*this_worker_id*/) {}

// -- QUEUE - MASTER COMMUNICATION --

void Messenger::send_from_queue_to_master() {}

void Messenger::send_from_master_to_queue()
{
   send_from_queue_to_master();
}

// for debugging
#define PROCESS_VAL(p) case(p): s = #p; break;

std::ostream& operator<<(std::ostream& out, const M2Q value){
   const char* s = 0;
   switch(value){
   PROCESS_VAL(M2Q::terminate);
   PROCESS_VAL(M2Q::enqueue);
   PROCESS_VAL(M2Q::retrieve);
   PROCESS_VAL(M2Q::update_real);
   }
   return out << s;
}

std::ostream& operator<<(std::ostream& out, const Q2M value){
   const char* s = 0;
   switch(value){
   PROCESS_VAL(Q2M::retrieve_rejected);
   PROCESS_VAL(Q2M::retrieve_accepted);
   }
   return out << s;
}

std::ostream& operator<<(std::ostream& out, const W2Q value){
   const char* s = 0;
   switch(value){
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
   PROCESS_VAL(Q2W::update_real);
   PROCESS_VAL(Q2W::result_received);
   }
   return out << s;
}

#undef PROCESS_VAL

} // namespace MultiProcess
} // namespace RooFit
