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
#include "RooFit/MultiProcess/Channel.h"
#include "RooFit/MultiProcess/Poller.h"

#include <iosfwd>
#include <string>
#include <vector>

namespace RooFit {
namespace MultiProcess {

// test messages
enum class X2X : int { ping = -1, pong = -2, initial_value = 0 };

class Messenger {
public:
   explicit Messenger(ProcessManager &process_manager);
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

   std::pair<Poller, std::size_t> create_queue_poller();
   std::pair<Poller, std::size_t> create_worker_poller();

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
   void publish_from_master_to_workers(T &&item);
   template <typename T, typename T2, typename... Ts>
   void publish_from_master_to_workers(T &&item, T2 &&item2, Ts &&...items);
   template <typename value_t>
   value_t receive_from_master_on_worker(bool *more = nullptr);

   template <typename T>
   void send_from_worker_to_master(T &&item);
   template <typename T, typename T2, typename... Ts>
   void send_from_worker_to_master(T &&item, T2 &&item2, Ts &&...items);
   template <typename value_t>
   value_t receive_from_worker_on_master(bool *more = nullptr);

   void test_receive(X2X expected_ping_value, test_rcv_pipes rcv_pipe, std::size_t worker_id);
   void test_send(X2X ping_value, test_snd_pipes snd_pipe, std::size_t worker_id);

private:
   void debug_print(std::string s);

   /// On master: pick the worker channel to receive the next message from.
   /// Continues an in-progress multipart message from the same worker;
   /// otherwise waits for any worker and picks one round-robin.
   Channel &select_worker_channel_on_master();
   void update_worker_channel_on_master(Channel &channel, bool more);

   // master-queue channel (on master and queue processes)
   Channel mq_;
   // queue-worker channels (all workers on the queue process, only the own
   // one on worker processes)
   std::vector<Channel> qw_;
   Channel this_worker_qw_;
   // master-worker channels, carrying both the state updates that were
   // previously published over PUB-SUB and the task results (all workers on
   // the master process, only the own one on worker processes)
   std::vector<Channel> mw_;
   Channel this_worker_mw_;

   // on master: bookkeeping for receiving from any worker
   Poller mw_poller_;
   Channel *mw_current_source_ = nullptr;
   std::size_t mw_next_poll_position_ = 0;
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
