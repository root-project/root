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

#include <csignal>  // sigprocmask etc
#include <cstdio>  // sprintf

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
   sigemptyset(&ppoll_sigmask);
   // create zmq connections (zmq context is automatically created in the ZeroMQSvc class and maintained as singleton)
   // and pollers where necessary
   try {
      if (process_manager.is_master()) {
         mq_push.reset(zmqSvc().socket_ptr(zmq::PUSH));
         mq_push->bind("ipc:///tmp/roofitMP_from_master_to_queue");

         mq_push_poller.register_socket(*mq_push, zmq::POLLOUT);

         mq_pull.reset(zmqSvc().socket_ptr(zmq::PULL));
         mq_pull->bind("ipc:///tmp/roofitMP_from_queue_to_master");

         mq_pull_poller.register_socket(*mq_pull, zmq::POLLIN);

         close_MQ_on_destruct_ = true;
      } else if (process_manager.is_queue()) {
         // first the queue-worker sockets
         // do resize instead of reserve so that the unique_ptrs are initialized
         // (to nullptr) so that we can do reset below, alternatively you can do
         // push/emplace_back with move or something
         qw_push.resize(process_manager.N_workers());
         qw_pull.resize(process_manager.N_workers());
         qw_push_poller.resize(process_manager.N_workers());
         qw_pull_poller.resize(process_manager.N_workers());
         for (std::size_t ix = 0; ix < process_manager.N_workers(); ++ix) {
            std::stringstream push_name, pull_name;
            // push
            qw_push[ix].reset(zmqSvc().socket_ptr(zmq::PUSH));
            push_name << "ipc:///tmp/roofitMP_from_queue_to_worker_" << ix;
            qw_push[ix]->bind(push_name.str());

            qw_push_poller[ix].register_socket(*qw_push[ix], zmq::POLLOUT);

            // pull
            qw_pull[ix].reset(zmqSvc().socket_ptr(zmq::PULL));
            pull_name << "ipc:///tmp/roofitMP_from_worker_" << ix << "_to_queue";
            qw_pull[ix]->bind(pull_name.str());

            qw_pull_poller[ix].register_socket(*qw_pull[ix], zmq::POLLIN);
         }

         // then the master-queue sockets
         mq_push.reset(zmqSvc().socket_ptr(zmq::PUSH));
         mq_push->connect("ipc:///tmp/roofitMP_from_queue_to_master");

         mq_push_poller.register_socket(*mq_push, zmq::POLLOUT);

         mq_pull.reset(zmqSvc().socket_ptr(zmq::PULL));
         mq_pull->connect("ipc:///tmp/roofitMP_from_master_to_queue");

         mq_pull_poller.register_socket(*mq_pull, zmq::POLLIN);

         close_MQ_on_destruct_ = true;
         close_QW_container_on_destruct_ = true;
      } else if (process_manager.is_worker()) {
         // we only need one queue-worker pipe on the worker
         qw_push_poller.resize(1);
         qw_pull_poller.resize(1);

         std::stringstream push_name, pull_name;
         // push
         this_worker_qw_push.reset(zmqSvc().socket_ptr(zmq::PUSH));
         push_name << "ipc:///tmp/roofitMP_from_worker_" << process_manager.worker_id() << "_to_queue";
         this_worker_qw_push->connect(push_name.str());

         qw_push_poller[0].register_socket(*this_worker_qw_push, zmq::POLLOUT);

         // pull
         this_worker_qw_pull.reset(zmqSvc().socket_ptr(zmq::PULL));
         pull_name << "ipc:///tmp/roofitMP_from_queue_to_worker_" << process_manager.worker_id();
         this_worker_qw_pull->connect(pull_name.str());

         qw_pull_poller[0].register_socket(*this_worker_qw_pull, zmq::POLLIN);

         close_this_QW_on_destruct_ = true;
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
   printf("somewhere %d...\n", getpid());
   if (close_MQ_on_destruct_) {
      try {
         mq_push.reset(nullptr);
         mq_pull.reset(nullptr);
      } catch (const std::exception& e) {
         std::cerr << "WARNING: something in Messenger dtor threw an exception! Original exception message:\n" << e.what() << std::endl;
      }
   }
   if (close_this_QW_on_destruct_) {
      this_worker_qw_push.reset(nullptr);
      this_worker_qw_pull.reset(nullptr);
   }
   if (close_QW_container_on_destruct_) {
      for (auto& socket : qw_push) {
         socket.reset(nullptr);
      }
      for (auto& socket : qw_pull) {
         socket.reset(nullptr);
      }
   }
   // NOTE: close_context didn't happen manually (from JobManager) on queue before, but probably did through this destructor (which I assumed wasn't called there during std::Exit_):
   zmqSvc().close_context();
   printf("...over the rainbow %d\n", getpid());
   // TODO: remove below comment if this new dtor works
   // the destructor is only used on the master process, so worker-queue
   // connections needn't be closed here; see documentation of JobManager
   // destructor
}


void Messenger::test_send(X2X ping_value, test_snd_pipes snd_pipe, std::size_t worker_id) {
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
}


void Messenger::test_receive(X2X expected_ping_value, test_rcv_pipes rcv_pipe, std::size_t worker_id) {
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
         printf("we get here... PID %d\n", getpid());
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
}


void Messenger::test_connections(const ProcessManager &process_manager) {
   // Before blocking SIGTERM, set the signal handler, so we can also check after blocking whether a signal occurred
   // In our case, we already set it in the ProcessManager after forking to the queue and worker processes.
   sigset_t sigmask;
   sigemptyset(&sigmask);
   sigaddset(&sigmask, SIGTERM);
   sigprocmask(SIG_BLOCK, &sigmask, &ppoll_sigmask);

   if (process_manager.is_master()) {
      test_send(X2X::ping, test_snd_pipes::M2Q, -1);
      test_receive(X2X::pong, test_rcv_pipes::fromQonM, -1);
      test_receive(X2X::ping, test_rcv_pipes::fromQonM, -1);
      test_send(X2X::pong, test_snd_pipes::M2Q, -1);
   } else if (process_manager.is_queue()) {
      ZeroMQPoller poller;
      std::size_t mq_index;
      std::tie(poller, mq_index) = create_queue_poller();

      for (std::size_t ix = 0; ix < process_manager.N_workers(); ++ix) {
         test_send(X2X::ping, test_snd_pipes::Q2W, ix);
      }

      int counter = 0;
      printf("poller.size(): %d\n", poller.size());
      while (!process_manager.sigterm_received() && (poller.size() > 0)) {
         // poll: wait until status change (-1: infinite timeout)
         printf("blurg %d\n", ++counter);
         std::vector<std::pair<size_t, int>> poll_result;
         try { // watch for zmq_error from ppoll caused by SIGTERM from master
            poll_result = poller.ppoll(-1, &ppoll_sigmask);
         } catch (zmq::error_t& e) {
            if ((e.num() == EINTR) && (ProcessManager::sigterm_received())) {
               break;
            } else if (e.num() == EAGAIN) {
               // This can happen from recv if ppoll initially gets a read-ready signal for a socket,
               // but the received data does not pass the checksum test, so the socket becomes unreadable
               // again or from non-blocking send if the socket becomes unwritable either due to the HWM
               // being reached or the socket not being connected (anymore). The latter case usually means
               // the connection has been severed from the other side, meaning it has probably been killed
               // and in that case the next ppoll call will probably also receive a SIGTERM, ending the
               // loop. In case something else is wrong, this message will print multiple times, which
               // should be taken as a cue for writing a bug report :)
               // TODO: handle this more rigorously
               std::cout << "EAGAIN in Queue::loop() (from either send or receive), continuing" << std::endl;
               continue;
            } else {
               throw;
            }
         }

         printf("blorzg %d\n", counter);

         // then process incoming messages from sockets
         for (auto readable_socket : poll_result) {
            // message comes from the master/queue socket (first element):
            if (readable_socket.first == mq_index) {
               printf("blarzeg1 %d\n", counter);
               test_receive(X2X::ping, test_rcv_pipes::fromMonQ, -1);
               printf("blarzeg2 %d\n", counter);
               test_send(X2X::pong, test_snd_pipes::Q2M, -1);
               printf("blarzeg3 %d\n", counter);
               test_send(X2X::ping, test_snd_pipes::Q2M, -1);
               printf("blarzeg4 %d\n", counter);
               test_receive(X2X::pong, test_rcv_pipes::fromMonQ, -1);
               printf("blarzeg5 %d\n", counter);
               poller.unregister_socket(*mq_pull);
               printf("poller.size(): %d\n", poller.size());
               printf("blarzeg6 %d\n", counter);
            } else { // from a worker socket
               printf("brimzongue1 %d\n", counter);
               // TODO: dangerous assumption for this_worker_id, may become invalid if we allow multiple queue_loops on the same process!
               auto this_worker_id = readable_socket.first - 1;  // TODO: replace with a more reliable lookup
               printf("brimzongue2 %d\n", counter);

               test_receive(X2X::pong, test_rcv_pipes::fromWonQ, this_worker_id);
               printf("brimzongue3 %d\n", counter);
               test_receive(X2X::ping, test_rcv_pipes::fromWonQ, this_worker_id);
               printf("brimzongue4 %d\n", counter);
               printf("sending to worker id %d\n", this_worker_id);
               test_send(X2X::pong, test_snd_pipes::Q2W, this_worker_id);
               printf("brimzongue5 %d\n", counter);

               poller.unregister_socket(*qw_pull[this_worker_id]);
               printf("poller.size(): %d\n", poller.size());
               printf("brimzongue6 %d\n", counter);
            }
         }
      }

   } else if (process_manager.is_worker()) {
      test_receive(X2X::ping, test_rcv_pipes::fromQonW, -1);
      test_send(X2X::pong, test_snd_pipes::W2Q, -1);
      test_send(X2X::ping, test_snd_pipes::W2Q, -1);
      printf("worker going into last receive, PID %d...\n", getpid());
      test_receive(X2X::pong, test_rcv_pipes::fromQonW, -1);
      printf("worker sent and received all tests, PID %d\n", getpid());
   } else {
      // should never get here
      throw std::runtime_error("Messenger::test_connections: I'm neither master, nor queue, nor a worker");
   }

   // clean up signal management modifications
   sigprocmask(SIG_SETMASK, &ppoll_sigmask, nullptr);
   if (process_manager.is_worker()) {
      printf("worker at end of test_connections, PID %d\n", getpid());
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
         printf("closing ZMQ context on PID %d\n", getpid());
         zmqSvc().close_context();
         printf("closed ZMQ context on PID %d\n", getpid());
      }
   } catch (const std::exception& e) {
      std::cerr << "WARNING: something in Messenger::terminate threw an exception! Original exception message:\n" << e.what() << std::endl;
   }
}


void Messenger::close_queue_worker_connections(const ProcessManager &process_manager, bool close_context) {
   if (process_manager.is_worker()) {
      this_worker_qw_push.reset(nullptr);
      this_worker_qw_pull.reset(nullptr);
      if (close_context) {
         zmqSvc().close_context();
      }
   } else if (process_manager.is_queue()) {
      for (std::size_t worker_ix = 0ul; worker_ix < process_manager.N_workers(); ++worker_ix) {
         qw_push[worker_ix].reset(nullptr);
         qw_pull[worker_ix].reset(nullptr);
      }
   }
}


std::pair<ZeroMQPoller, std::size_t> Messenger::create_queue_poller() {
   ZeroMQPoller poller;
   std::size_t mq_index = poller.register_socket(*mq_pull, zmq::POLLIN);
   for (auto &s : qw_pull) {
      poller.register_socket(*s, zmq::POLLIN);
   }
   return {std::move(poller), mq_index};
}


ZeroMQPoller Messenger::create_worker_poller() {
   ZeroMQPoller poller;
   poller.register_socket(*this_worker_qw_pull, zmq::POLLIN);
   return poller;
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

void Messenger::send_from_master_to_queue() {}

void Messenger::set_send_flag(int flag) {
   if (flag == 0 || flag == ZMQ_DONTWAIT || flag == ZMQ_SNDMORE || flag == (ZMQ_DONTWAIT | ZMQ_SNDMORE)) {
      send_flag = flag;
   } else {
      throw std::runtime_error("in Messenger::set_send_flag: trying to set illegal flag, see zmq_send API for allowed flags");
   }
}


// for debugging
#define PROCESS_VAL(p) case(p): s = #p; break;

std::ostream& operator<<(std::ostream& out, const M2Q value){
   std::string s;
   switch(value){
   PROCESS_VAL(M2Q::terminate);
   PROCESS_VAL(M2Q::enqueue);
   PROCESS_VAL(M2Q::retrieve);
   PROCESS_VAL(M2Q::update_real);
   PROCESS_VAL(M2Q::update_bool);
   default: s = std::to_string(static_cast<int>(value));
   }
   return out << s;
}

std::ostream& operator<<(std::ostream& out, const Q2M value){
   std::string s;
   switch(value){
   PROCESS_VAL(Q2M::retrieve_rejected);
   PROCESS_VAL(Q2M::retrieve_accepted);
   PROCESS_VAL(Q2M::retrieve_later);
   default: s = std::to_string(static_cast<int>(value));
   }
   return out << s;
}

std::ostream& operator<<(std::ostream& out, const W2Q value){
   std::string s;
   switch(value){
   PROCESS_VAL(W2Q::dequeue);
   PROCESS_VAL(W2Q::send_result);
   default: s = std::to_string(static_cast<int>(value));
   }
   return out << s;
}

std::ostream& operator<<(std::ostream& out, const Q2W value){
   std::string s;
   switch(value){
   PROCESS_VAL(Q2W::terminate);
   PROCESS_VAL(Q2W::dequeue_rejected);
   PROCESS_VAL(Q2W::dequeue_accepted);
   PROCESS_VAL(Q2W::update_real);
   PROCESS_VAL(Q2W::result_received);
   PROCESS_VAL(Q2W::update_bool);
   default: s = std::to_string(static_cast<int>(value));
   }
   return out << s;
}

std::ostream& operator<<(std::ostream& out, const X2X value){
   std::string s;
   switch(value){
   PROCESS_VAL(X2X::ping);
   PROCESS_VAL(X2X::pong);
   default: s = std::to_string(static_cast<int>(value));
   }
   return out << s;
}

#undef PROCESS_VAL

void Messenger::debug_print(std::string s)
{
   printf("%s\n", s.c_str());
//   std::cerr << s << std::endl;
}

} // namespace MultiProcess
} // namespace RooFit
