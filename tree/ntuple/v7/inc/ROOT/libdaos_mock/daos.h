/*
 * (C) Copyright 2016-2018 Intel Corporation.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * \file
 *
 * This file is a reduced version of `daos_xxx.h` headers that provides (simplified) declarations for use in
 * libdaos_mock.
 */

#ifndef __DAOS_H__
#define __DAOS_H__
extern "C" {

//////////////////////////////////////////////////////////////////////////////// daos_types.h

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <uuid/uuid.h>
#include <ctype.h>

/** iovec for memory buffer */
typedef struct {
	void		*iov_buf;
	size_t		iov_buf_len;
	size_t		iov_len;
} d_iov_t;

/** Scatter/gather list for memory buffers */
typedef struct {
	uint32_t	sg_nr;
	uint32_t	sg_nr_out;
	d_iov_t		*sg_iovs;
} d_sg_list_t;

static inline void d_iov_set(d_iov_t *iov, void *buf, size_t size)
{
	iov->iov_buf = buf;
	iov->iov_len = iov->iov_buf_len = size;
}

typedef uint64_t	daos_size_t;

/** Generic handle for various DAOS components like container, object, etc. */
typedef struct {
	uint64_t	cookie;
} daos_handle_t;

typedef enum {
   DAOS_EQR_COMPLETED = (1),
   DAOS_EQR_WAITING = (1 << 1),
   DAOS_EQR_ALL = (DAOS_EQR_COMPLETED | DAOS_EQR_WAITING),
} daos_eq_query_t;

#define DAOS_HDL_INVAL	((daos_handle_t){0})
#define DAOS_TX_NONE	DAOS_HDL_INVAL

#define DAOS_PC_RO	(1U << 0)
#define DAOS_PC_RW	(1U << 1)
#define DAOS_PC_EX	(1U << 2)

typedef d_iov_t daos_key_t;

/** Event and event queue */
typedef struct daos_event {
	int			ev_error;
	struct {
		uint64_t	space[19];
	}			ev_private;
	uint64_t		ev_debug;
} daos_event_t;

/** Wait for completion event forever */
#define DAOS_EQ_WAIT            -1
/** Always return immediately */
#define DAOS_EQ_NOWAIT          0


//////////////////////////////////////////////////////////////////////////////// daos_event.h

int daos_eq_create(daos_handle_t *eqh);
int daos_eq_destroy(daos_handle_t eqh, int flags);
int daos_eq_poll(daos_handle_t eqh, int wait_running, int64_t timeout, unsigned int nevents, daos_event_t **events);

int daos_event_test(daos_event_t *ev, int64_t timeout, bool *flag);
int daos_event_parent_barrier(daos_event_t *ev);
int daos_event_init(daos_event_t *ev, daos_handle_t eqh, daos_event_t *parent);
int daos_event_fini(daos_event_t *ev);

//////////////////////////////////////////////////////////////////////////////// daos_obj_class.h


/** Predefined object classes */
enum {
	OC_UNKNOWN	= 0,

	/** Replicated object class which is extremely scalable for fetch. */
	OC_RP_XSF	= 80,

	/** Object classes with explicit layout */
	OC_S1		= 200,
	OC_S2,
	OC_S4,
	OC_S8,
	OC_S16,
	OC_S32,
	OC_S64,
	OC_S128,
	OC_S256,
	OC_S512,
	OC_S1K,
	OC_S2K,
	OC_S4K,
	OC_S8K,
	OC_SX,

	/** Class ID equal or higher than this is reserved */
	OC_RESERVED		= (1U << 10),
};

typedef uint16_t daos_oclass_id_t;
typedef uint16_t daos_oclass_hints_t;

int daos_oclass_name2id(const char *name);
int daos_oclass_id2name(daos_oclass_id_t oc_id, char *name);


//////////////////////////////////////////////////////////////////////////////// daos_obj.h


typedef struct {
	uint64_t	lo;
	uint64_t	hi;
} daos_obj_id_t;

#define DAOS_OBJ_NIL ((daos_obj_id_t){0})

#define OID_FMT_INTR_BITS 32 // 32 bits for DAOS internal use
#define OID_FMT_TYPE_BITS 8
#define OID_FMT_CLASS_BITS 8
#define OID_FMT_META_BITS 16

#define OID_FMT_TYPE_SHIFT (64 - OID_FMT_TYPE_BITS)
#define OID_FMT_CLASS_SHIFT (OID_FMT_TYPE_SHIFT - OID_FMT_CLASS_BITS)
#define OID_FMT_META_SHIFT (OID_FMT_CLASS_SHIFT - OID_FMT_META_BITS)

/// DAOS object type
enum daos_otype_t {
   DAOS_OT_MULTI_HASHED = 0, // default: multi-level KV with hashed [ad]keys
   DAOS_OT_DKEY_UINT64 = 2,
   DAOS_OT_AKEY_UINT64 = 3,
   DAOS_OT_MULTI_UINT64 = 4,
   DAOS_OT_ARRAY = 11,
   DAOS_OT_ARRAY_BYTE = 13,
   DAOS_OT_MAX = 13,
};

static inline bool daos_otype_t_is_valid(enum daos_otype_t type)
{
   return type <= DAOS_OT_MAX;
}

enum {
   DAOS_COND_DKEY_FETCH = (1 << 3),
   DAOS_COND_AKEY_FETCH = (1 << 6),
};

/** Object open modes */
enum {
   DAOS_OO_RO = (1 << 1),
   DAOS_OO_RW = (1 << 2),
};

typedef struct {
	uint64_t	rx_idx;
	uint64_t	rx_nr;
} daos_recx_t;

/** Type of the value accessed in an IOD */
typedef enum {
	DAOS_IOD_SINGLE		= 1,
} daos_iod_type_t;

typedef struct {
	daos_key_t		iod_name;
	daos_iod_type_t		iod_type;
	daos_size_t		iod_size;
	unsigned int		iod_nr;
	daos_recx_t		*iod_recxs;
} daos_iod_t;

typedef struct {
	char		unused; // silence [-Wextern-c-compat]
} daos_iom_t;

enum {
	/** Any record size, it is used by fetch */
	DAOS_REC_ANY		= 0,
};

/// Flags for oclass hints
enum {
   // OC Redundancy flags
   DAOS_OCH_RDD_DEF = (1 << 0), // Default: RF prop
   // OC Sharding flags
   DAOS_OCH_SHD_DEF = (1 << 4), // Default: MAX for array & flat KV, else 1
};

int daos_obj_generate_oid(daos_handle_t coh, daos_obj_id_t *oid, enum daos_otype_t type, daos_oclass_id_t cid,
                          daos_oclass_hints_t hints, uint32_t args);
int daos_obj_open(daos_handle_t coh, daos_obj_id_t oid, unsigned int mode, daos_handle_t *oh, daos_event_t *ev);
int daos_obj_close(daos_handle_t oh, daos_event_t *ev);
int daos_obj_fetch(daos_handle_t oh, daos_handle_t th, uint64_t flags,
	       daos_key_t *dkey, unsigned int nr, daos_iod_t *iods,
	       d_sg_list_t *sgls, daos_iom_t *ioms, daos_event_t *ev);
int daos_obj_update(daos_handle_t oh, daos_handle_t th, uint64_t flags,
		daos_key_t *dkey, unsigned int nr, daos_iod_t *iods,
		d_sg_list_t *sgls, daos_event_t *ev);


//////////////////////////////////////////////////////////////////////////////// daos_prop.h

/** daos properties, for pool or container */
typedef struct {
	char		unused; // silence [-Wextern-c-compat]
} daos_prop_t;

#define DAOS_PROP_LABEL_MAX_LEN (127)
#define DAOS_PROP_MAX_LABEL_BUF_LEN (DAOS_PROP_LABEL_MAX_LEN + 1)
#define DAOS_UUID_STR_SIZE (37) // 36 + 1 for '\0'

static inline bool daos_label_is_valid(const char * /*label*/)
{
   return true;
}

//////////////////////////////////////////////////////////////////////////////// daos_cont.h


#define DAOS_COO_RO	(1U << 0)
#define DAOS_COO_RW	(1U << 1)

/** Container information */
typedef struct {
   uuid_t ci_uuid;
} daos_cont_info_t;

int daos_cont_create_with_label(daos_handle_t poh, const char *label, daos_prop_t *cont_prop, uuid_t *uuid,
                                daos_event_t *ev);
int daos_cont_open(daos_handle_t poh, const char *uuid, unsigned int flags, daos_handle_t *coh, daos_cont_info_t *info,
                   daos_event_t *ev);
int daos_cont_close(daos_handle_t coh, daos_event_t *ev);

//////////////////////////////////////////////////////////////////////////////// daos_pool.h


/** Storage pool */
typedef struct {
   uuid_t pi_uuid;
} daos_pool_info_t;

int daos_pool_connect(const char *pool, const char *grp, unsigned int flags, daos_handle_t *poh, daos_pool_info_t *info,
                      daos_event_t *ev);
int daos_pool_disconnect(daos_handle_t poh, daos_event_t *ev);

//////////////////////////////////////////////////////////////////////////////// daos_errno.h

#define DER_ERR_GURT_BASE 1000
#define DER_INVAL (DER_ERR_GURT_BASE + 3)
#define DER_EXIST (DER_ERR_GURT_BASE + 4)

const char *d_errstr(int rc);


//////////////////////////////////////////////////////////////////////////////// daos.h


int daos_init(void);
int daos_fini(void);
}
#endif /* __DAOS_H__ */
