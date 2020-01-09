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
#include "../inc/RooFit/MultiProcess/Messenger_decl.h"

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
         mq_push.reset(zmqSvc().socket_ptr(zmq::PUSH));
         mq_push.reset(zmqSvc().socket_ptr(zmq::PUSH));
         mq_push->bind("ipc:///tmp/roofitMP_from_master_to_queue");
         mq_pull.reset(zmqSvc().socket_ptr(zmq::PULL));
         mq_pull.reset(zmqSvc().socket_ptr(zmq::PULL));
         mq_pull->bind("ipc:///tmp/roofitMP_from_queue_to_master");
      } else if (process_manager.is_queue()) {
         // first the queue-worker sockets
         // do resize instead of reserve so that the unique_ptrs are initialized
         // (to nullptr) so that we can do reset below, alternatively you can do
         // push/emplace_back with move or something
         qw_push.resize(process_manager.N_workers());
         qw_pull.resize(process_manager.N_workers());
         for (std::size_t ix = 0; ix < process_manager.N_workers(); ++ix) {
            std::stringstream push_name, pull_name;
            // push
            qw_push[ix].reset(zmqSvc().socket_ptr(zmq::PUSH));
            push_name << "ipc:///tmp/roofitMP_from_queue_to_worker_" << ix;
            qw_push[ix]->bind(push_name.str());
            // pull
            qw_pull[ix].reset(zmqSvc().socket_ptr(zmq::PULL));
            pull_name << "ipc:///tmp/roofitMP_from_worker_" << ix << "_to_queue";
            qw_pull[ix]->bind(pull_name.str());
         }
         // then the master-queue sockets
         mq_push.reset(zmqSvc().socket_ptr(zmq::PUSH));
         mq_push->connect("ipc:///tmp/roofitMP_from_queue_to_master");
         mq_pull.reset(zmqSvc().socket_ptr(zmq::PULL));
         mq_pull->connect("ipc:///tmp/roofitMP_from_master_to_queue");
      } else if (process_manager.is_worker()) {
         std::stringstream push_name, pull_name;
         // push
         this_worker_qw_push.reset(zmqSvc().socket_ptr(zmq::PUSH));
         push_name << "ipc:///tmp/roofitMP_from_worker_" << process_manager.worker_id() << "_to_queue";
         this_worker_qw_push->connect(push_name.str());
         // pull
         this_worker_qw_pull.reset(zmqSvc().socket_ptr(zmq::PULL));
         pull_name << "ipc:///tmp/roofitMP_from_queue_to_worker_" << process_manager.worker_id();
         this_worker_qw_pull->connect(pull_name.str());
      } else {
         // should never get here
         throw std::runtime_error("Messenger ctor: I'm neither master, nor queue, nor a worker");
      }
   } catch (zmq::error_t &e) {
      std::cerr << e.what() << " -- errnum: " << e.num() << std::endl;
      throw;
   };
}

Messenger::~Messenger() {
   close_master_queue_connection(true);
   // the destructor is only used on the master process, so worker-queue
   // connections needn't be closed here; see documentation of JobManager
   // destructor
}


void Messenger::test_send(ZmqLingeringSocketPtr<> &socket, X2X ping_value, test_snd_pipes snd_pipe, std::size_t worker_id) {
   int timeout_ms = 5000;
   int reset_timeout = socket->getsockopt<int>(ZMQ_SNDTIMEO);
   socket->setsockopt(ZMQ_SNDTIMEO, &timeout_ms, sizeof(timeout_ms));

   try {
      switch (snd_pipe) {
      case test_snd_pipes::M2Q: {
         send_from_master_to_queue(ping_value);
         break;
      }
      case test_snd_pipes::Q2M : {
         send_from_queue_to_master(ping_value);
         break;
      }
      case test_snd_pipes::Q2W: {
         send_from_queue_to_worker(worker_id, ping_value);
         break;
      }
      case test_snd_pipes::W2Q: {
         send_from_worker_to_queue(ping_value);
         break;
      }
      }
   } catch (zmq::error_t &e) {
      if (e.num() == EAGAIN) {
         throw std::runtime_error("Messenger::test_connections: SEND over master-queue connection timed out!");
      } else {
         throw;
      }
   }

   socket->setsockopt(ZMQ_SNDTIMEO, &reset_timeout, sizeof(reset_timeout));
}


void Messenger::test_receive(ZmqLingeringSocketPtr<> &socket, X2X expected_ping_value, test_rcv_pipes rcv_pipe, std::size_t worker_id) {
   int timeout_ms = 5000;
   int reset_timeout = socket->getsockopt<int>(ZMQ_RCVTIMEO);
   socket->setsockopt(ZMQ_RCVTIMEO, &timeout_ms, sizeof(timeout_ms));

   X2X handshake;

   try {
      switch (rcv_pipe) {
      case test_rcv_pipes::fromMonQ: {
         handshake = receive_from_master_on_queue<X2X>();
         break;
      }
      case test_rcv_pipes::fromQonM : {
         handshake = receive_from_queue_on_master<X2X>();
         break;
      }
      case test_rcv_pipes::fromQonW: {
         handshake = receive_from_queue_on_worker<X2X>();
         break;
      }
      case test_rcv_pipes::fromWonQ: {
         handshake = receive_from_worker_on_queue<X2X>(worker_id);
         break;
      }
      }

   } catch (zmq::error_t &e) {
      if (e.num() == EAGAIN) {
         throw std::runtime_error("Messenger::test_connections: RECEIVE over master-queue connection timed out!");
      } else {
         throw;
      }
   }

   if (handshake != expected_ping_value) {
      throw std::runtime_error("Messenger::test_connections: RECEIVE over master-queue connection failed, did not receive pong!");
   }

   socket->setsockopt(ZMQ_RCVTIMEO, &reset_timeout, sizeof(reset_timeout));
}


void Messenger::test_connections(const ProcessManager &process_manager) {
   process_manager.identify_processes();
   if (process_manager.is_master()) {
      std::cout << "testing Messenger connections on master" << std::endl;
      test_send(mq_push, X2X::ping, test_snd_pipes::M2Q, -1);
      test_receive(mq_pull, X2X::pong, test_rcv_pipes::fromQonM, -1);
      test_receive(mq_pull, X2X::ping, test_rcv_pipes::fromQonM, -1);
      test_send(mq_push, X2X::pong, test_snd_pipes::M2Q, -1);
      std::cout << "DONE testing Messenger connections on master" << std::endl;
   } else if (process_manager.is_queue()) {
      std::cout << "testing Messenger connections on queue" << std::endl;
      ZeroMQPoller poller;
      std::size_t mq_index;
      std::tie(poller, mq_index) = create_poller();

      for (std::size_t ix = 0; ix < process_manager.N_workers(); ++ix) {
         test_send(qw_push[ix], X2X::ping, test_snd_pipes::Q2W, ix);
      }

      while (!process_manager.sigterm_received() && (poller.size() > 0)) {
         // poll: wait until status change (-1: infinite timeout)
         std::cout << "queue polling" << std::endl;
         auto poll_result = poller.poll(-1);
         std::cout << "queue got poll result with size " << poll_result.size() << std::endl;
         // then process incoming messages from sockets
         for (auto readable_socket : poll_result) {
            std::cout << "readable_socket.second: " << readable_socket.second << std::endl;
            // message comes from the master/queue socket (first element):
            if (readable_socket.first == mq_index) {
               std::cout << "queue doing master" << std::endl;
               test_receive(mq_pull, X2X::ping, test_rcv_pipes::fromMonQ, -1);
               test_send(mq_push, X2X::pong, test_snd_pipes::Q2M, -1);
               test_send(mq_push, X2X::ping, test_snd_pipes::Q2M, -1);
               test_receive(mq_pull, X2X::pong, test_rcv_pipes::fromMonQ, -1);
               poller.unregister_socket(*mq_pull);
               std::cout << "queue done with master" << std::endl;
            } else { // from a worker socket
               // TODO: dangerous assumption for this_worker_id, may become invalid if we allow multiple queue_loops on the same process!
               auto this_worker_id = readable_socket.first - 1;  // TODO: replace with a more reliable lookup
               std::cout << "queue doing worker " << this_worker_id << std::endl;

               test_receive(qw_pull[this_worker_id], X2X::pong, test_rcv_pipes::fromWonQ, this_worker_id);
               test_receive(qw_pull[this_worker_id], X2X::ping, test_rcv_pipes::fromWonQ, this_worker_id);
               test_send(qw_push[this_worker_id], X2X::pong, test_snd_pipes::Q2W, this_worker_id);

               poller.unregister_socket(*qw_pull[this_worker_id]);
               std::cout << "queue done with worker " << this_worker_id << std::endl;
            }
         }
      }
      std::cout << "DONE testing Messenger connections on queue" << std::endl;
   } else if (process_manager.is_worker()) {
      std::cout << "testing Messenger connections on worker " << process_manager.worker_id() << std::endl;
      test_receive(this_worker_qw_pull, X2X::ping, test_rcv_pipes::fromQonW, -1);
      test_send(this_worker_qw_push, X2X::pong, test_snd_pipes::W2Q, -1);
      test_send(this_worker_qw_push, X2X::ping, test_snd_pipes::W2Q, -1);
      test_receive(this_worker_qw_pull, X2X::pong, test_rcv_pipes::fromQonW, -1);
      std::cout << "DONE testing Messenger connections on worker " << process_manager.worker_id() << std::endl;
   } else {
      // should never get here
      throw std::runtime_error("Messenger::test_connections: I'm neither master, nor queue, nor a worker");
   }
}


void Messenger::close_master_queue_connection(bool close_context) noexcept {
   // this function is called from the Messenger destructor on the master process
   // and from JobManager::activate on the queue process before _Exiting that process
   // so we need not check for those processes here (also, we can't on master, because
   // there we are already in the process of destroying the JobManager _instance, so
   // our link to the process_manager is gone and we can't pass it as an argument to
   // the destructor)
   try {
      mq_push.reset(nullptr);
      mq_pull.reset(nullptr);
      if (close_context) {
         zmqSvc().close_context();
      }
   } catch (const std::exception& e) {
      std::cerr << "WARNING: something in Messenger::terminate threw an exception! Original exception message:\n" << e.what() << std::endl;
   }
}


void Messenger::close_queue_worker_connections(bool close_context) {
   if (JobManager::instance()->process_manager().is_worker()) {
      this_worker_qw_push.reset(nullptr);
      this_worker_qw_pull.reset(nullptr);
      if (close_context) {
         zmqSvc().close_context();
      }
   } else if (JobManager::instance()->process_manager().is_queue()) {
      for (std::size_t worker_ix = 0ul; worker_ix < JobManager::instance()->process_manager().N_workers(); ++worker_ix) {
         qw_push[worker_ix].reset(nullptr);
         qw_pull[worker_ix].reset(nullptr);
      }
   }
}


std::pair<ZeroMQPoller, std::size_t> Messenger::create_poller() {
   ZeroMQPoller poller;
   std::size_t mq_index = poller.register_socket(*mq_pull, zmq::POLLIN);
   for (auto &s : qw_pull) {
      poller.register_socket(*s, zmq::POLLIN);
   }
   return {std::move(poller), mq_index};
}


//bool Messenger::is_initialized() const
//{
//   if (JobManager::instance()->process_manager().is_worker()) {
//      return static_cast<bool>(this_worker_qw_socket);
//   } else if (JobManager::instance()->process_manager().is_queue()) {
//      return static_cast<bool>(qw_sockets[0]) && static_cast<bool>(mq_socket);
//   } else if (JobManager::instance()->process_manager().is_master()) {
//      return static_cast<bool>(mq_socket);
//   } else {
//      return false;
//   }
//}

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
   PROCESS_VAL(Q2M::retrieve_later);
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
