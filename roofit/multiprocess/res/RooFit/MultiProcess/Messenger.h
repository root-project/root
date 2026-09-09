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

#include <sstream>
#include <unistd.h> // getpid

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

   send_item(this_worker_qw_, item, /*more=*/false);
   send_from_worker_to_queue(items...);
}

template <typename value_t>
value_t Messenger::receive_from_worker_on_queue(std::size_t this_worker_id)
{
   auto value = receive_item<value_t>(qw_[this_worker_id]);

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

   send_item(qw_[this_worker_id], item, /*more=*/false);
   send_from_queue_to_worker(this_worker_id, items...);
}

template <typename value_t>
value_t Messenger::receive_from_queue_on_worker()
{
   auto value = receive_item<value_t>(this_worker_qw_);

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

   send_item(mq_, item, /*more=*/false);
   send_from_queue_to_master(items...);
}

template <typename value_t>
value_t Messenger::receive_from_queue_on_master()
{
   auto value = receive_item<value_t>(mq_);

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

   send_item(mq_, item, /*more=*/false);
   send_from_master_to_queue(items...);
}

template <typename value_t>
value_t Messenger::receive_from_master_on_queue()
{
   auto value = receive_item<value_t>(mq_);

#ifndef NDEBUG
   std::stringstream ss;
   ss << "PID " << getpid() << " receives M2Q " << value;
   debug_print(ss.str());
#endif

   return value;
}

// -- MASTER - WORKER COMMUNICATION --

/// specialization that sends the final part of a message
template <typename T>
void Messenger::publish_from_master_to_workers(T &&item)
{
#ifndef NDEBUG
   std::stringstream ss;
   ss << "PID " << getpid() << " sends M2W " << item;
   debug_print(ss.str());
#endif

   for (auto &channel : mw_) {
      send_item(channel, item, /*more=*/false);
   }
}

/// specialization that sends the first parts of multipart messages
template <typename T, typename T2, typename... Ts>
void Messenger::publish_from_master_to_workers(T &&item, T2 &&item2, Ts &&...items)
{
#ifndef NDEBUG
   std::stringstream ss;
   ss << "PID " << getpid() << " sends M2W " << item;
   debug_print(ss.str());
#endif

   for (auto &channel : mw_) {
      send_item(channel, item, /*more=*/true);
   }
   publish_from_master_to_workers(std::forward<T2>(item2), std::forward<Ts>(items)...);
}

template <typename value_t>
value_t Messenger::receive_from_master_on_worker(bool *more)
{
   auto value = receive_item<value_t>(this_worker_mw_, more);

#ifndef NDEBUG
   std::stringstream ss;
   ss << "PID " << getpid() << " receives M2W " << value;
   debug_print(ss.str());
#endif

   return value;
}

/// specialization that sends the final part of a message
template <typename T>
void Messenger::send_from_worker_to_master(T &&item)
{
#ifndef NDEBUG
   std::stringstream ss;
   ss << "PID " << getpid() << " sends W2M " << item;
   debug_print(ss.str());
#endif

   send_item(this_worker_mw_, item, /*more=*/false);
}

/// specialization that sends the first parts of multipart messages
template <typename T, typename T2, typename... Ts>
void Messenger::send_from_worker_to_master(T &&item, T2 &&item2, Ts &&...items)
{
#ifndef NDEBUG
   std::stringstream ss;
   ss << "PID " << getpid() << " sends W2M " << item;
   debug_print(ss.str());
#endif

   send_item(this_worker_mw_, item, /*more=*/true);
   send_from_worker_to_master(std::forward<T2>(item2), std::forward<Ts>(items)...);
}

template <typename value_t>
value_t Messenger::receive_from_worker_on_master(bool *more)
{
   Channel &channel = select_worker_channel_on_master();
   bool more_parts = false;
   auto value = receive_item<value_t>(channel, &more_parts);
   update_worker_channel_on_master(channel, more_parts);
   if (more) {
      *more = more_parts;
   }

#ifndef NDEBUG
   std::stringstream ss;
   ss << "PID " << getpid() << " receives W2M " << value;
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
