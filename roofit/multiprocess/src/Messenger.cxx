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

#include "RooFit/MultiProcess/Messenger.h"
#include "RooFit/MultiProcess/util.h"

#include <iostream>
#include <stdexcept>

namespace RooFit {
namespace MultiProcess {

/** \class Messenger
 *
 * \brief Manages the interprocess communication channels and wraps send and receive calls
 *
 * This class is used for all interprocess communication between the master,
 * queue and worker processes. The communication runs over pipes built on
 * socketpair(), which are created in the ProcessManager before forking, so
 * that all processes inherit their ends of the connected channels; see
 * Channel for the wire format.
 *
 * Several channels connect the processes for different purposes:
 * - The master and queue processes share a channel that is mainly used for
 *   sending tasks to the queue from master.
 * - The queue process shares a channel with each worker process. These are
 *   used by the workers to obtain tasks from the queue.
 * - The master shares a channel with each worker process. The master -> worker
 *   direction carries state updates (previously published over a ZeroMQ
 *   PUB-SUB socket) and the worker -> master direction carries back task
 *   results, which the master receives in 'JobManager::retrieve()'.
 *
 * @param process_manager ProcessManager instance which manages the master,
 *                        queue and worker processes that we want to set up
 *                        communication for in this Messenger.
 */

Messenger::Messenger(ProcessManager &process_manager)
{
   // Claim the channel ends for this process type from the ProcessManager,
   // which created them before forking. The channels are connected from
   // birth, so no connection handshake is necessary.
   if (process_manager.is_master()) {
      mq_ = Channel{process_manager.claim_mq_fd()};
      mw_.reserve(process_manager.N_workers());
      for (std::size_t ix = 0; ix < process_manager.N_workers(); ++ix) {
         mw_.emplace_back(process_manager.claim_mw_fd(ix));
      }
      for (auto &channel : mw_) {
         mw_poller_.register_channel(channel);
      }
   } else if (process_manager.is_queue()) {
      mq_ = Channel{process_manager.claim_mq_fd()};
      qw_.reserve(process_manager.N_workers());
      for (std::size_t ix = 0; ix < process_manager.N_workers(); ++ix) {
         qw_.emplace_back(process_manager.claim_qw_fd(ix));
      }
   } else if (process_manager.is_worker()) {
      this_worker_qw_ = Channel{process_manager.claim_qw_fd(process_manager.worker_id())};
      this_worker_mw_ = Channel{process_manager.claim_mw_fd(process_manager.worker_id())};
   } else {
      // should never get here
      throw std::runtime_error("Messenger ctor: I'm neither master, nor queue, nor a worker");
   }
}

Messenger::~Messenger() = default;

void Messenger::test_send(X2X ping_value, test_snd_pipes snd_pipe, std::size_t worker_id)
{
   switch (snd_pipe) {
   case test_snd_pipes::M2Q: {
      send_from_master_to_queue(ping_value);
      break;
   }
   case test_snd_pipes::Q2M: {
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
}

void Messenger::test_receive(X2X expected_ping_value, test_rcv_pipes rcv_pipe, std::size_t worker_id)
{
   X2X handshake = X2X::initial_value;

   try {
      switch (rcv_pipe) {
      case test_rcv_pipes::fromMonQ: {
         handshake = receive_from_master_on_queue<X2X>();
         break;
      }
      case test_rcv_pipes::fromQonM: {
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
   } catch (ppoll_error_t &) {
      throw std::runtime_error("SIGTERM received in test_receive, aborting\n");
   }

   if (handshake != expected_ping_value) {
      throw std::runtime_error(
         "Messenger::test_connections: RECEIVE over connection failed, did not receive expected value!");
   }
}

/// \brief Test whether the channels between all processes are working
///
/// \param process_manager ProcessManager object used to instantiate this object. Used to identify which process we are
/// running on and hence which channels need to be tested.
void Messenger::test_connections(const ProcessManager &process_manager)
{
   if (process_manager.is_master()) {
      test_receive(X2X::ping, test_rcv_pipes::fromQonM, -1);
      test_send(X2X::pong, test_snd_pipes::M2Q, -1);
      test_send(X2X::ping, test_snd_pipes::M2Q, -1);
      // make sure to always receive last on master, so that master knows when queue is done,
      // which means workers are done as well, so if master is done everything is done:
      test_receive(X2X::pong, test_rcv_pipes::fromQonM, -1);
   } else if (process_manager.is_queue()) {
      Poller poller;
      std::size_t mq_index;
      std::tie(poller, mq_index) = create_queue_poller();

      for (std::size_t ix = 0; ix < process_manager.N_workers(); ++ix) {
         test_send(X2X::ping, test_snd_pipes::Q2W, ix);
      }
      test_send(X2X::ping, test_snd_pipes::Q2M, -1);

      while (!process_manager.sigterm_received() && (poller.size() > 0)) {
         // poll: wait until status change (-1: infinite timeout)
         std::vector<std::size_t> poll_result;
         bool abort;
         std::tie(poll_result, abort) = careful_poll(poller);
         if (abort)
            break;

         // then process incoming messages from the channels
         for (auto readable_index : poll_result) {
            // message comes from the master/queue channel (first element):
            if (readable_index == mq_index) {
               test_receive(X2X::pong, test_rcv_pipes::fromMonQ, -1);
               test_receive(X2X::ping, test_rcv_pipes::fromMonQ, -1);
               poller.unregister_channel(mq_);
            } else {                                     // from a worker channel
               auto this_worker_id = readable_index - 1; // by construction of the queue poller
               test_receive(X2X::pong, test_rcv_pipes::fromWonQ, this_worker_id);
               test_receive(X2X::ping, test_rcv_pipes::fromWonQ, this_worker_id);
               test_send(X2X::pong, test_snd_pipes::Q2W, this_worker_id);

               poller.unregister_channel(qw_[this_worker_id]);
            }
         }
      }
      test_send(X2X::pong, test_snd_pipes::Q2M, -1);

   } else if (process_manager.is_worker()) {
      test_receive(X2X::ping, test_rcv_pipes::fromQonW, -1);
      test_send(X2X::pong, test_snd_pipes::W2Q, -1);
      test_send(X2X::ping, test_snd_pipes::W2Q, -1);
      test_receive(X2X::pong, test_rcv_pipes::fromQonW, -1);
   } else {
      // should never get here
      throw std::runtime_error("Messenger::test_connections: I'm neither master, nor queue, nor a worker");
   }
}

/// Helper function that creates a poller for Queue::loop()
std::pair<Poller, std::size_t> Messenger::create_queue_poller()
{
   Poller poller;
   std::size_t mq_index = poller.register_channel(mq_);
   for (auto &channel : qw_) {
      poller.register_channel(channel);
   }
   return {std::move(poller), mq_index};
}

/// Helper function that creates a poller for worker_loop()
std::pair<Poller, std::size_t> Messenger::create_worker_poller()
{
   Poller poller;
   poller.register_channel(this_worker_qw_);
   std::size_t mw_index = poller.register_channel(this_worker_mw_);
   return {std::move(poller), mw_index};
}

Channel &Messenger::select_worker_channel_on_master()
{
   // continue receiving the parts of an in-progress multipart message from
   // the same worker (multipart messages must arrive as one unit, like with
   // the ZeroMQ sockets used before)
   if (mw_current_source_ != nullptr) {
      return *mw_current_source_;
   }
   auto readable = mw_poller_.poll(-1);
   // rotate over the workers for fairness, like a ZeroMQ PULL socket would
   for (std::size_t offset = 0; offset < mw_.size(); ++offset) {
      std::size_t candidate = (mw_next_poll_position_ + offset) % mw_.size();
      for (std::size_t index : readable) {
         if (index == candidate) {
            mw_next_poll_position_ = (candidate + 1) % mw_.size();
            return mw_[candidate];
         }
      }
   }
   // cannot happen: poll(-1) always returns at least one readable channel
   throw std::logic_error("Messenger::select_worker_channel_on_master: poll returned no readable channels");
}

void Messenger::update_worker_channel_on_master(Channel &channel, bool more)
{
   mw_current_source_ = more ? &channel : nullptr;
}

// -- WORKER - QUEUE COMMUNICATION --

void Messenger::send_from_worker_to_queue() {}

void Messenger::send_from_queue_to_worker(std::size_t /*this_worker_id*/) {}

// -- QUEUE - MASTER COMMUNICATION --

void Messenger::send_from_queue_to_master() {}

void Messenger::send_from_master_to_queue() {}

// for debugging
#define PROCESS_VAL(p) \
   case (p): s = #p; break;

std::ostream &operator<<(std::ostream &out, const M2Q value)
{
   std::string s;
   switch (value) {
      PROCESS_VAL(M2Q::enqueue);
   default: s = std::to_string(static_cast<int>(value));
   }
   return out << s;
}

std::ostream &operator<<(std::ostream &out, const W2Q value)
{
   std::string s;
   switch (value) {
      PROCESS_VAL(W2Q::dequeue);
   default: s = std::to_string(static_cast<int>(value));
   }
   return out << s;
}

std::ostream &operator<<(std::ostream &out, const Q2W value)
{
   std::string s;
   switch (value) {
      PROCESS_VAL(Q2W::dequeue_rejected);
      PROCESS_VAL(Q2W::dequeue_accepted);
   default: s = std::to_string(static_cast<int>(value));
   }
   return out << s;
}

std::ostream &operator<<(std::ostream &out, const X2X value)
{
   std::string s;
   switch (value) {
      PROCESS_VAL(X2X::ping);
      PROCESS_VAL(X2X::pong);
   default: s = std::to_string(static_cast<int>(value));
   }
   return out << s;
}

#undef PROCESS_VAL

/// Function called from send and receive template functions in debug builds
/// used to monitor the messages that are going to be sent or are received.
/// By defining this in the implementation file, compilation is a lot faster
/// during debugging of Messenger or communication protocols.
void Messenger::debug_print(std::string /*s*/)
{
   // print 's' when debugging
}

} // namespace MultiProcess
} // namespace RooFit
