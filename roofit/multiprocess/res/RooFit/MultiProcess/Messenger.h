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
#ifndef ROOT_ROOFIT_MultiProcess_Messenger
#define ROOT_ROOFIT_MultiProcess_Messenger

#include "RooFit/MultiProcess/Messenger_decl.h"

#ifdef NDEBUG
#undef NDEBUG
#define turn_NDEBUG_back_on
#endif

namespace RooFit {
namespace MultiProcess {

// -- WORKER - QUEUE COMMUNICATION --

template <typename T, typename... Ts>
void Messenger::send_from_worker_to_queue(T item, Ts... items)
{
#ifndef NDEBUG
   std::stringstream ss;
   ss << "PID " << getpid() << " sends W2Q " << item;
   debug_print(ss.str());
#endif

   zmqSvc().send(*this_worker_qw_push_, item, send_flag_);
   //      if (sizeof...(items) > 0) {  // this will only work with if constexpr, c++17
   send_from_worker_to_queue(items...);
}

template <typename value_t>
value_t Messenger::receive_from_worker_on_queue(std::size_t this_worker_id)
{
   qw_pull_poller_[this_worker_id].ppoll(-1, &ppoll_sigmask);
   auto value = zmqSvc().receive<value_t>(*qw_pull_[this_worker_id], zmq::recv_flags::dontwait);

#ifndef NDEBUG
   std::stringstream ss;
   ss << "PID " << getpid() << " receives W(" << this_worker_id << ")2Q " << value;
   debug_print(ss.str());
#endif

   return value;
}

template <typename T, typename... Ts>
void Messenger::send_from_queue_to_worker(std::size_t this_worker_id, T item, Ts... items)
{
#ifndef NDEBUG
   std::stringstream ss;
   ss << "PID " << getpid() << " sends Q2W(" << this_worker_id << ") " << item;
   debug_print(ss.str());
#endif

   zmqSvc().send(*qw_push_[this_worker_id], item, send_flag_);
   //      if (sizeof...(items) > 0) {  // this will only work with if constexpr, c++17
   send_from_queue_to_worker(this_worker_id, items...);
}

template <typename value_t>
value_t Messenger::receive_from_queue_on_worker()
{
   qw_pull_poller_[0].ppoll(-1, &ppoll_sigmask);
   auto value = zmqSvc().receive<value_t>(*this_worker_qw_pull_, zmq::recv_flags::dontwait);

#ifndef NDEBUG
   std::stringstream ss;
   ss << "PID " << getpid() << " receives Q2W " << value;
   debug_print(ss.str());
#endif

   return value;
}

// -- QUEUE - MASTER COMMUNICATION --

template <typename T, typename... Ts>
void Messenger::send_from_queue_to_master(T item, Ts... items)
{
#ifndef NDEBUG
   std::stringstream ss;
   ss << "PID " << getpid() << " sends Q2M " << item;
   debug_print(ss.str());
#endif

   zmqSvc().send(*mq_push_, item, send_flag_);
   //      if (sizeof...(items) > 0) {  // this will only work with if constexpr, c++17
   send_from_queue_to_master(items...);
}

template <typename value_t>
value_t Messenger::receive_from_queue_on_master()
{
   mq_pull_poller_.ppoll(-1, &ppoll_sigmask);
   auto value = zmqSvc().receive<value_t>(*mq_pull_, zmq::recv_flags::dontwait);

#ifndef NDEBUG
   std::stringstream ss;
   ss << "PID " << getpid() << " receives Q2M " << value;
   debug_print(ss.str());
#endif

   return value;
}

template <typename T, typename... Ts>
void Messenger::send_from_master_to_queue(T item, Ts... items)
{
#ifndef NDEBUG
   std::stringstream ss;
   ss << "PID " << getpid() << " sends M2Q " << item;
   debug_print(ss.str());
#endif

   zmqSvc().send(*mq_push_, item, send_flag_);
   //      if (sizeof...(items) > 0) {  // this will only work with if constexpr, c++17
   send_from_master_to_queue(items...);
}

template <typename value_t>
value_t Messenger::receive_from_master_on_queue()
{
   mq_pull_poller_.ppoll(-1, &ppoll_sigmask);
   auto value = zmqSvc().receive<value_t>(*mq_pull_, zmq::recv_flags::dontwait);

#ifndef NDEBUG
   std::stringstream ss;
   ss << "PID " << getpid() << " receives M2Q " << value;
   debug_print(ss.str());
#endif

   return value;
}

// -- MASTER - WORKER COMMUNICATION --

/// specialization that sends the final message
template <typename T>
void Messenger::publish_from_master_to_workers(T&& item)
{
#ifndef NDEBUG
   std::stringstream ss;
   ss << "PID " << getpid() << " sends M2W " << item;
   debug_print(ss.str());
#endif

   zmqSvc().send(*mw_pub_, std::forward<T>(item), send_flag_);
}

/// specialization that queues first parts of multipart messages
template <typename T, typename T2, typename... Ts>
void Messenger::publish_from_master_to_workers(T&& item, T2&& item2, Ts&&... items)
{
#ifndef NDEBUG
   std::stringstream ss;
   ss << "PID " << getpid() << " sends M2W " << item;
   debug_print(ss.str());
#endif

   zmqSvc().send(*mw_pub_, std::forward<T>(item), send_flag_ | zmq::send_flags::sndmore);
   publish_from_master_to_workers(std::forward<T2>(item2), std::forward<Ts>(items)...);
}

template <typename value_t>
value_t Messenger::receive_from_master_on_worker(bool *more)
{
   mw_sub_poller_.ppoll(-1, &ppoll_sigmask);
   auto value = zmqSvc().receive<value_t>(*mw_sub_, zmq::recv_flags::dontwait, more);

#ifndef NDEBUG
   std::stringstream ss;
   ss << "PID " << getpid() << " receives M2W " << value;
   debug_print(ss.str());
#endif

   return value;
}

template <typename T, typename... Ts>
void Messenger::send_from_worker_to_master(T item, Ts... items)
{
#ifndef NDEBUG
   std::stringstream ss;
   ss << "PID " << getpid() << " sends M2W " << item;
   debug_print(ss.str());
#endif

   zmqSvc().send(*wm_push_, item, send_flag_);
   //      if (sizeof...(items) > 0) {  // this will only work with if constexpr, c++17
   send_from_worker_to_master(items...);
}

template <typename value_t>
value_t Messenger::receive_from_worker_on_master()
{
   wm_pull_poller_.ppoll(-1, &ppoll_sigmask);
   auto value = zmqSvc().receive<value_t>(*wm_pull_, zmq::recv_flags::dontwait);

#ifndef NDEBUG
   std::stringstream ss;
   ss << "PID " << getpid() << " receives M2W " << value;
   debug_print(ss.str());
#endif

   return value;
}

} // namespace MultiProcess
} // namespace RooFit

#ifdef turn_NDEBUG_back_on
#define NDEBUG
#undef turn_NDEBUG_back_on
#endif

#endif // ROOT_ROOFIT_MultiProcess_Messenger
