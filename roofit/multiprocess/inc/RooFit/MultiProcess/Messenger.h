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
#ifndef ROOT_ROOFIT_MultiProcess_Messenger
#define ROOT_ROOFIT_MultiProcess_Messenger

#include "RooFit/MultiProcess/Messenger_decl.h"
#include "Messenger_decl.h"

namespace RooFit {
namespace MultiProcess {

// -- WORKER - QUEUE COMMUNICATION --

template <typename T, typename... Ts>
void Messenger::send_from_worker_to_queue(T item, Ts... items)
{
   try {
      zmqSvc().send(*this_worker_qw_socket, item);
   } catch (zmq::error_t &e) {
      std::cerr << e.what() << " -- errnum: " << e.num() << std::endl;
      throw;
   };
   //      if (sizeof...(items) > 0) {  // this will only work with if constexpr, c++17
   send_from_worker_to_queue(items...);
}

template <typename value_t>
value_t Messenger::receive_from_worker_on_queue(std::size_t this_worker_id)
{
   try {
      auto value = zmqSvc().receive<value_t>(*qw_sockets[this_worker_id]);
      return value;
   } catch (zmq::error_t &e) {
      std::cerr << e.what() << " -- errnum: " << e.num() << std::endl;
      throw;
   };
}

template <typename T, typename... Ts>
void Messenger::send_from_queue_to_worker(std::size_t this_worker_id, T item, Ts... items)
{
   try {
      zmqSvc().send(*qw_sockets[this_worker_id], item);
   } catch (zmq::error_t &e) {
      std::cerr << e.what() << " -- errnum: " << e.num() << std::endl;
      throw;
   };
   //      if (sizeof...(items) > 0) {  // this will only work with if constexpr, c++17
   send_from_queue_to_worker(this_worker_id, items...);
}

template <typename value_t>
value_t Messenger::receive_from_queue_on_worker()
{
   try {
      auto value = zmqSvc().receive<value_t>(*this_worker_qw_socket);
      return value;
   } catch (zmq::error_t &e) {
      std::cerr << e.what() << " -- errnum: " << e.num() << std::endl;
      throw;
   };
}


// -- QUEUE - MASTER COMMUNICATION --

template <typename T, typename... Ts>
void Messenger::send_from_queue_to_master(T item, Ts... items)
{
   try {
      zmqSvc().send(*mq_socket, item);
   } catch (zmq::error_t &e) {
      std::cerr << e.what() << " -- errnum: " << e.num() << std::endl;
      throw;
   };
   //      if (sizeof...(items) > 0) {  // this will only work with if constexpr, c++17
   send_from_queue_to_master(items...);
}

template <typename value_t>
value_t Messenger::receive_from_queue_on_master()
{
   try {
      auto value = zmqSvc().receive<value_t>(*mq_socket);
      return value;
   } catch (zmq::error_t &e) {
      std::cerr << e.what() << " -- errnum: " << e.num() << std::endl;
      throw;
   };
}

template <typename T, typename... Ts>
void Messenger::send_from_master_to_queue(T item, Ts... items)
{
   send_from_queue_to_master(item, items...);
}

template <typename value_t>
value_t Messenger::receive_from_master_on_queue()
{
   return receive_from_queue_on_master<value_t>();
}

} // namespace MultiProcess
} // namespace RooFit

#endif // ROOT_ROOFIT_MultiProcess_Messenger
