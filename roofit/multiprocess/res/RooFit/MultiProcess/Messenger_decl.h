/*
 * Project: RooFit
 * Authors:
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl
 *   IP, Inti Pelupessy, Netherlands eScience Center, i.pelupessy@esciencecenter.nl
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */
#ifndef ROOT_ROOFIT_MultiProcess_Messenger_decl
#define ROOT_ROOFIT_MultiProcess_Messenger_decl

#include "RooFit/MultiProcess/ProcessManager.h"
#include "RooFit_ZMQ/ZeroMQSvc.h"
#include "RooFit_ZMQ/ZeroMQPoller.h"

#include <iosfwd>
#include <vector>
#include <csignal> // sigprocmask, sigset_t, etc
#include <string>

namespace RooFit {
namespace MultiProcess {

void set_socket_immediate(ZmqLingeringSocketPtr<> &socket);

// test messages
enum class X2X : int { ping = -1, pong = -2, initial_value = 0 };

class Messenger {
public:
   explicit Messenger(const ProcessManager &process_manager);
   ~Messenger();

   void test_connections(const ProcessManager &process_manager);

   enum class test_snd_pipes {
      M2Q,
      Q2M,
      Q2W,
      W2Q,
   };

   enum class test_rcv_pipes {
      fromQonM,
      fromMonQ,
      fromWonQ,
      fromQonW,
   };

   std::pair<ZeroMQPoller, std::size_t> create_queue_poller();
   std::pair<ZeroMQPoller, std::size_t> create_worker_poller();

   // -- WORKER - QUEUE COMMUNICATION --

   void send_from_worker_to_queue();
   template <typename T, typename... Ts>
   void send_from_worker_to_queue(T item, Ts... items);
   template <typename value_t>
   value_t receive_from_worker_on_queue(std::size_t this_worker_id);
   void send_from_queue_to_worker(std::size_t this_worker_id);
   template <typename T, typename... Ts>
   void send_from_queue_to_worker(std::size_t this_worker_id, T item, Ts... items);
   template <typename value_t>
   value_t receive_from_queue_on_worker();

   // -- QUEUE - MASTER COMMUNICATION --

   void send_from_queue_to_master();

   template <typename T, typename... Ts>
   void send_from_queue_to_master(T item, Ts... items);
   template <typename value_t>
   value_t receive_from_queue_on_master();
   void send_from_master_to_queue();

   template <typename T, typename... Ts>
   void send_from_master_to_queue(T item, Ts... items);
   template <typename value_t>
   value_t receive_from_master_on_queue();

   // -- MASTER - WORKER COMMUNICATION --

   template <typename T>
   void publish_from_master_to_workers(T&& item);
   template <typename T, typename T2, typename... Ts>
   void publish_from_master_to_workers(T&& item, T2&& item2, Ts&&... items);
   template <typename value_t>
   value_t receive_from_master_on_worker(bool *more = nullptr);

   void send_from_worker_to_master();
   template <typename T, typename... Ts>
   void send_from_worker_to_master(T item, Ts... items);
   template <typename value_t>
   value_t receive_from_worker_on_master();

   void test_receive(X2X expected_ping_value, test_rcv_pipes rcv_pipe, std::size_t worker_id);
   void test_send(X2X ping_value, test_snd_pipes snd_pipe, std::size_t worker_id);

   sigset_t ppoll_sigmask;

   void set_send_flag(zmq::send_flags flag);

private:
   void debug_print(std::string s);

   template<class T>
   void bindAddr(T & socket, std::string && addr) {
     bound_ipc_addresses_.emplace_back(addr);
     socket->bind(bound_ipc_addresses_.back());
   }

   // push
   std::vector<ZmqLingeringSocketPtr<>> qw_push_;
   ZmqLingeringSocketPtr<> this_worker_qw_push_;
   ZmqLingeringSocketPtr<> mq_push_;
   // pollers for all push sockets
   std::vector<ZeroMQPoller> qw_push_poller_;
   ZeroMQPoller mq_push_poller_;
   // pull
   std::vector<ZmqLingeringSocketPtr<>> qw_pull_;
   ZmqLingeringSocketPtr<> this_worker_qw_pull_;
   ZmqLingeringSocketPtr<> mq_pull_;
   // pollers for all pull sockets
   std::vector<ZeroMQPoller> qw_pull_poller_;
   ZeroMQPoller mq_pull_poller_;

   // publish/subscribe sockets for parameter updating from master to workers
   ZmqLingeringSocketPtr<> mw_pub_;
   ZmqLingeringSocketPtr<> mw_sub_;
   ZeroMQPoller mw_sub_poller_;
   // push/pull sockets for result retrieving from workers on master
   ZmqLingeringSocketPtr<> wm_push_;
   ZmqLingeringSocketPtr<> wm_pull_;
   ZeroMQPoller wm_pull_poller_;

   // destruction flags to distinguish between different process-type setups:
   bool close_MQ_on_destruct_ = false;
   bool close_this_QW_on_destruct_ = false;
   bool close_QW_container_on_destruct_ = false;

   zmq::send_flags send_flag_ = zmq::send_flags::none;

   std::vector<std::string> bound_ipc_addresses_;
};

// Messages from master to queue
enum class M2Q : int {
   enqueue = 10,
};

// Messages from worker to queue
enum class W2Q : int { dequeue = 30 };

// Messages from queue to worker
enum class Q2W : int {
   dequeue_rejected = 40,
   dequeue_accepted = 41,
};

// stream output operators for debugging
std::ostream &operator<<(std::ostream &out, const M2Q value);
std::ostream &operator<<(std::ostream &out, const Q2W value);
std::ostream &operator<<(std::ostream &out, const W2Q value);
std::ostream &operator<<(std::ostream &out, const X2X value);

} // namespace MultiProcess
} // namespace RooFit

#endif // ROOT_ROOFIT_MultiProcess_Messenger_decl
