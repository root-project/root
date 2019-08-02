/*
 * IOchain - Distribute a chain of dependant IO events amoung threads.
 *
 * This file is part of Bitshuffle
 * Author: Kiyoshi Masui <kiyo@physics.ubc.ca>
 * Website: http://www.github.com/kiyo-masui/bitshuffle
 * Created: 2014
 *
 * See LICENSE file for details about copyright and rights to use.
 *
 */

#include <stdlib.h>
#include "Bitshuffle_iochain.h"


void ioc_init(ioc_chain *C, const void *in_ptr_0, void *out_ptr_0) {
#ifdef _OPENMP
    omp_init_lock(&C->next_lock);
    for (size_t ii = 0; ii < IOC_SIZE; ii ++) {
        omp_init_lock(&(C->in_pl[ii].lock));
        omp_init_lock(&(C->out_pl[ii].lock));
    }
#endif
    C->next = 0;
    C->in_pl[0].ptr = in_ptr_0;
    C->out_pl[0].ptr = out_ptr_0;
}


void ioc_destroy(ioc_chain *C) {
#ifdef _OPENMP
    omp_destroy_lock(&C->next_lock);
    for (size_t ii = 0; ii < IOC_SIZE; ii ++) {
        omp_destroy_lock(&(C->in_pl[ii].lock));
        omp_destroy_lock(&(C->out_pl[ii].lock));
    }
#endif
}


const void * ioc_get_in(ioc_chain *C, size_t *this_iter) {
#ifdef _OPENMP
    omp_set_lock(&C->next_lock);
    #pragma omp flush
#endif
    *this_iter = C->next;
    C->next ++;
#ifdef _OPENMP
    omp_set_lock(&(C->in_pl[*this_iter % IOC_SIZE].lock));
    omp_set_lock(&(C->in_pl[(*this_iter + 1) % IOC_SIZE].lock));
    omp_set_lock(&(C->out_pl[(*this_iter + 1) % IOC_SIZE].lock));
    omp_unset_lock(&C->next_lock);
#endif
    return C->in_pl[*this_iter % IOC_SIZE].ptr;
}


void ioc_set_next_in(ioc_chain *C, size_t* this_iter, void* in_ptr) {
    C->in_pl[(*this_iter + 1) % IOC_SIZE].ptr = in_ptr;
#ifdef _OPENMP
    omp_unset_lock(&(C->in_pl[(*this_iter + 1) % IOC_SIZE].lock));
#endif
}


void * ioc_get_out(ioc_chain *C, size_t *this_iter) {
#ifdef _OPENMP
    omp_set_lock(&(C->out_pl[(*this_iter) % IOC_SIZE].lock));
    #pragma omp flush
#endif
    void *out_ptr = C->out_pl[*this_iter % IOC_SIZE].ptr;
#ifdef _OPENMP
    omp_unset_lock(&(C->out_pl[(*this_iter) % IOC_SIZE].lock));
#endif
    return out_ptr;
}


void ioc_set_next_out(ioc_chain *C, size_t *this_iter, void* out_ptr) {
    C->out_pl[(*this_iter + 1) % IOC_SIZE].ptr = out_ptr;
#ifdef _OPENMP
    omp_unset_lock(&(C->out_pl[(*this_iter + 1) % IOC_SIZE].lock));
    // *in_pl[this_iter]* lock released at the end of the iteration to avoid being
    // overtaken by previous threads and having *out_pl[this_iter]* corrupted.
    // Especially worried about thread 0, iteration 0.
    omp_unset_lock(&(C->in_pl[(*this_iter) % IOC_SIZE].lock));
#endif
}
