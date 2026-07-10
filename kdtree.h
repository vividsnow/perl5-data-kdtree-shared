/*
 * kdtree.h -- Shared-memory k-d tree for Linux
 *
 * Spatial index over points in up to 16 dimensions: nearest-neighbour,
 * k-nearest, axis-aligned box (range), and radius (ball) queries.  Points are
 * appended in O(1) and a balanced tree is bulk-built by median split on the
 * first query after any insert, so query recursion stays O(log n) deep
 * regardless of insertion order.  The points and tree live in a shared mapping
 * so several processes build and query one index; a write-preferring futex
 * rwlock with reader-slot dead-process recovery guards mutation, and queries
 * take only the read lock once the tree is built.
 *
 * Layout: Header -> reader_slots[1024] -> nodes[capacity] -> build_idx[capacity]
 */

#ifndef KD_H
#define KD_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>
#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <sys/syscall.h>
#include <linux/futex.h>
#include <pthread.h>

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#error "kdtree.h: requires little-endian architecture"
#endif


/* ================================================================
 * Constants
 * ================================================================ */

#define KD_MAGIC        0x5254444B  /* KDTree */
#define KD_VERSION      1
#define KD_ERR_BUFLEN   256
#ifndef KD_READER_SLOTS
#define KD_READER_SLOTS 1024         /* max concurrent reader processes for dead-process recovery */
#endif
#define KD_MIN_DIMS     1
#define KD_MAX_DIMS     16               /* max spatial dimensions */
#define KD_MIN_CAP      1
#define KD_MAX_CAP      0x1000000U        /* 2^24 points cap (index fits uint32, < KD_NIL) */
#define KD_NIL          0xFFFFFFFFU       /* empty child / root sentinel */
/* A balanced bulk-built tree over <= 2^24 points is <= ~25 deep; this cap is a
 * runaway guard so a Layer-B-corrupted link chain (a cycle, or a degenerate
 * chain of valid indices) cannot recurse a query into a stack overflow. */
#define KD_MAX_DEPTH    96

#define KD_ERR(fmt, ...) do { if (errbuf) snprintf(errbuf, KD_ERR_BUFLEN, fmt, ##__VA_ARGS__); } while (0)

/* ================================================================
 * Structs
 * ================================================================ */

/* Per-process slot for dead-process recovery.  Each shared rwlock counter
 * (the main rwlock-reader count, rwlock_waiters, rwlock_writers_waiting)
 * is mirrored here so a wrlock timeout can attribute and reverse a dead
 * process's contribution instead of waiting for the slow per-op timeout
 * drain. */
typedef struct {
    uint32_t pid;            /* 0 = unclaimed */
    uint32_t subcount;       /* in-flight rdlock acquisitions for this process */
    uint32_t waiters_parked; /* contribution to hdr->rwlock_waiters         */
    uint32_t writers_parked; /* contribution to hdr->rwlock_writers_waiting */
} KdReaderSlot;

struct KdHeader {
    uint32_t magic, version;          /* 0,4 */
    uint32_t dims;                    /* 8   number of spatial dimensions */
    uint32_t capacity;                /* 12  max points */
    uint64_t count;                   /* 16  points inserted */
    uint32_t root;                    /* 24  root node index, or KD_NIL when empty */
    uint32_t dirty;                   /* 28  1 if the tree needs a (re)build before querying */
    uint64_t node_stride;             /* 32  bytes per node (dims*8 + 16) */
    uint64_t nodes_off;               /* 40  offset of the node array */
    uint64_t idx_off;                 /* 48  offset of the build scratch (capacity uint32) */
    uint64_t total_size;              /* 56 */
    uint64_t reader_slots_off;        /* 64 */
    uint32_t rwlock;                  /* 72 */
    uint32_t rwlock_waiters;          /* 76 */
    uint32_t rwlock_writers_waiting;  /* 80 */
    uint32_t slotless_readers;  /* live readers holding the lock with no reader-slot */
    uint64_t stat_ops;                /* 88 */
    uint8_t  _pad[160];               /* 96..255 */
};
typedef struct KdHeader KdHeader;

_Static_assert(sizeof(KdHeader) == 256, "KdHeader must be 256 bytes");

/* Node layout (variable, stride = dims*8 + 16, always 8-aligned):
 *   [dims doubles: coords][uint64 payload id][uint32 left][uint32 right]      */

/* ---- Process-local handle ---- */

typedef struct KdHandle {
    KdHeader     *hdr;
    KdReaderSlot *reader_slots;  /* KD_READER_SLOTS entries */
    void         *base;          /* mmap base */
    uint64_t      nodes_off;     /* validated store offsets, cached: never re-read from the peer-writable header */
    uint64_t      idx_off;
    uint64_t      node_stride;   /* cached */
    uint32_t      dims;          /* cached */
    uint32_t      capacity;      /* cached */
    size_t        mmap_size;
    char         *path;          /* backing file path (strdup'd) */
    int           backing_fd;    /* memfd or reopened-fd to close on destroy, -1 for file/anon */
    uint32_t      my_slot_idx;   /* UINT32_MAX if all slots taken (no recovery for this handle) */
    uint32_t      cached_pid;    /* getpid() cached at last slot claim */
    uint32_t      cached_fork_gen; /* kd_fork_gen value at last slot claim */
    uint32_t slotless_held; /* rwlock read-locks held with no reader-slot */
} KdHandle;

/* ================================================================
 * Futex-based write-preferring read-write lock
 * with reader-slot dead-process recovery
 * ================================================================ */

#define KD_RWLOCK_SPIN_LIMIT 32
#define KD_LOCK_TIMEOUT_SEC  2  /* FUTEX_WAIT timeout for stale lock detection */

static inline void kd_rwlock_spin_pause(void) {
#if defined(__x86_64__) || defined(__i386__)
    __asm__ volatile("pause" ::: "memory");
#elif defined(__aarch64__)
    __asm__ volatile("yield" ::: "memory");
#else
    __asm__ volatile("" ::: "memory");
#endif
}

/* Extract writer PID from rwlock value (lower 31 bits when write-locked). */
#define KD_RWLOCK_WRITER_BIT 0x80000000U
#define KD_RWLOCK_PID_MASK   0x7FFFFFFFU
#define KD_RWLOCK_WR(pid)    (KD_RWLOCK_WRITER_BIT | ((uint32_t)(pid) & KD_RWLOCK_PID_MASK))

/* Check if a PID is alive. Returns 1 if alive or unknown, 0 if definitely dead. */
/* Liveness via kill(pid,0). NOTE: cannot detect PID reuse -- if a dead
 * lock-holder's PID is recycled to an unrelated live process before recovery
 * runs, this reports "alive" and that slot's orphaned contribution is not
 * reclaimed until the recycled process exits. Robust detection would require
 * a per-slot process-start-time epoch (a header-layout/version change).
 * Documented under "Crash Safety" in the POD. */
static inline int kd_pid_alive(uint32_t pid) {
    if (pid == 0) return 1; /* no owner recorded, assume alive */
    return !(kill((pid_t)pid, 0) == -1 && errno == ESRCH);
}

/* Force-recover a stale write lock left by a dead process.
 * CAS to OUR pid to hold the lock while fixing shared state, then release.
 * Using our pid (not a bare WRITER_BIT sentinel) means a subsequent
 * recovering process can detect and re-recover if we crash mid-recovery. */
static inline void kd_recover_stale_lock(KdHandle *h, uint32_t observed_rwlock) {
    KdHeader *hdr = h->hdr;
    uint32_t mypid = KD_RWLOCK_WR((uint32_t)getpid());
    if (!__atomic_compare_exchange_n(&hdr->rwlock, &observed_rwlock,
            mypid, 0, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED))
        return;
    /* We now hold the write lock as mypid.  No additional shared state needs
     * repair here (this module has no seqlock); just release the lock. */
    __atomic_store_n(&hdr->rwlock, 0, __ATOMIC_RELEASE);
    if (__atomic_load_n(&hdr->rwlock_waiters, __ATOMIC_RELAXED) > 0)
        syscall(SYS_futex, &hdr->rwlock, FUTEX_WAKE, INT_MAX, NULL, NULL, 0);
}

static const struct timespec kd_lock_timeout = { KD_LOCK_TIMEOUT_SEC, 0 };

/* Process-global fork-generation counter.  Incremented in the pthread_atfork
 * child callback so every open handle detects a fork transition on the next
 * lock call without paying a getpid() syscall on the hot path. */
static uint32_t kd_fork_gen = 1;
static pthread_once_t kd_atfork_once = PTHREAD_ONCE_INIT;
static void kd_on_fork_child(void) {
    __atomic_add_fetch(&kd_fork_gen, 1, __ATOMIC_RELAXED);
}
static void kd_atfork_init(void) {
    pthread_atfork(NULL, NULL, kd_on_fork_child);
}

/* Ensure this process owns a reader slot.  Called from the lock helpers so
 * that fork()'d children pick up their own slot lazily instead of sharing
 * the parent's.  Hot-path is a single relaxed load + compare; only on a
 * fork-generation mismatch do we touch getpid() and scan slots. */
static inline void kd_claim_reader_slot(KdHandle *h) {
    uint32_t cur_gen = __atomic_load_n(&kd_fork_gen, __ATOMIC_RELAXED);
    if (__builtin_expect(cur_gen == h->cached_fork_gen && h->my_slot_idx != UINT32_MAX, 1))
        return;
    /* Cold path -- register the atfork hook once per process, then claim. */
    pthread_once(&kd_atfork_once, kd_atfork_init);
    /* Re-read after pthread_once: kd_on_fork_child may have bumped it. */
    cur_gen = __atomic_load_n(&kd_fork_gen, __ATOMIC_RELAXED);
    uint32_t now_pid = (uint32_t)getpid();
    h->cached_pid = now_pid;
    if (cur_gen != h->cached_fork_gen) h->slotless_held = 0;  /* fork: child holds none of the parent's slotless read locks */
    h->cached_fork_gen = cur_gen;
    h->my_slot_idx = UINT32_MAX;
    uint32_t start = now_pid % KD_READER_SLOTS;
    for (uint32_t i = 0; i < KD_READER_SLOTS; i++) {
        uint32_t s = (start + i) % KD_READER_SLOTS;
        uint32_t expected = 0;
        if (__atomic_compare_exchange_n(&h->reader_slots[s].pid,
                &expected, now_pid, 0,
                __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) {
            /* Zero all mirror fields, not just subcount: a SIGKILL'd
             * predecessor may have left waiters_parked/writers_parked
             * non-zero, and kd_recover_dead_readers won't drain them
             * once we own the slot (the CAS expects the dead PID). */
            __atomic_store_n(&h->reader_slots[s].subcount, 0, __ATOMIC_RELAXED);
            __atomic_store_n(&h->reader_slots[s].waiters_parked, 0, __ATOMIC_RELAXED);
            __atomic_store_n(&h->reader_slots[s].writers_parked, 0, __ATOMIC_RELAXED);
            h->my_slot_idx = s;
            return;
        }
    }
    /* Table full -- leave my_slot_idx = UINT32_MAX so we silently skip
     * tracking for this handle (lock still works; just no recovery). */
}

/* Atomically subtract `sub` from a counter, capped at 0 (never underflows). */
static inline void kd_atomic_sub_cap(uint32_t *p, uint32_t sub) {
    if (!sub) return;
    uint32_t cur = __atomic_load_n(p, __ATOMIC_RELAXED);
    for (;;) {
        uint32_t want = (cur > sub) ? cur - sub : 0;
        if (__atomic_compare_exchange_n(p, &cur, want,
                1, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
            return;
    }
}

/* Try to claim a dead slot (CAS pid -> 0) and drain its parked-waiter
 * contributions back to the global counters.  A no-op if the slot was stolen
 * by another recoverer or had no waiter contribution to drain.
 *
 * Note: subcount/waiters_parked/writers_parked are NOT zeroed here.
 * Between our CAS and a follow-up store, a new process could claim the
 * slot and start populating these fields -- our stores would clobber its
 * state.  kd_claim_reader_slot zeros all three on every claim, so
 * leaving stale values is harmless. */
static inline void kd_drain_dead_slot(KdHandle *h, uint32_t i, uint32_t pid) {
    KdHeader *hdr = h->hdr;
    uint32_t expected = pid;
    /* ACQ_REL on success: RELEASE publishes pid=0 to other observers;
     * ACQUIRE syncs us with prior writes from the dead process to
     * waiters_parked/writers_parked.  On weakly-ordered archs (aarch64)
     * a plain RELAXED load before the CAS could miss those writes;
     * loading them after the CAS keeps them inside the acquire window. */
    if (!__atomic_compare_exchange_n(&h->reader_slots[i].pid, &expected, 0,
            0, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
        return;
    uint32_t wp    = __atomic_load_n(&h->reader_slots[i].waiters_parked, __ATOMIC_RELAXED);
    uint32_t writp = __atomic_load_n(&h->reader_slots[i].writers_parked, __ATOMIC_RELAXED);
    if (wp)    kd_atomic_sub_cap(&hdr->rwlock_waiters, wp);
    if (writp) kd_atomic_sub_cap(&hdr->rwlock_writers_waiting, writp);
}

/* Scan reader slots for dead-process recovery.
 *
 * For each dead PID with non-zero contributions to the shared rwlock,
 * rwlock_waiters, or rwlock_writers_waiting counters, drain its share back
 * out so live processes don't have to wait for the slow per-op timeout
 * decrement to drain it for them.
 *
 * For the main rwlock counter we use the "no live reader holds -> force-
 * reset to 0" trick (precise) because per-process attribution of the
 * subcount is racy across the inc-counter-then-inc-subcount window. */
static inline void kd_recover_dead_readers(KdHandle *h) {
    if (!h->reader_slots) return;
    KdHeader *hdr = h->hdr;
    int any_live_reader = 0;
    int found_dead_reader = 0;

    /* Pass 1: classify slots.  Slots with dead pid and sc == 0 (no rwlock
     * contribution to lose) are wiped immediately to free the slot for
     * future claimants and drain any orphan parked-waiter counters.  Slots
     * with dead pid and sc > 0 are left intact in this pass: if force-
     * reset cannot fire (because a live reader is concurrently present),
     * wiping the dead slot would lose the only record of its orphan
     * rwlock contribution and strand writers permanently once the live
     * reader releases. */
    for (uint32_t i = 0; i < KD_READER_SLOTS; i++) {
        uint32_t pid = __atomic_load_n(&h->reader_slots[i].pid, __ATOMIC_ACQUIRE);
        if (pid == 0) continue;
        uint32_t sc = __atomic_load_n(&h->reader_slots[i].subcount, __ATOMIC_RELAXED);
        if (kd_pid_alive(pid)) {
            if (sc > 0) any_live_reader = 1;
            continue;
        }
        if (sc > 0) { found_dead_reader = 1; continue; }
        kd_drain_dead_slot(h, i, pid);
    }

    /* Pass 2: only if force-reset will fire.  Issue the rwlock force-
     * reset CAS FIRST, while the window since pass 1's last scan is
     * still narrow (a handful of instructions, as in the original
     * single-pass code).  A new reader that started rdlock between
     * pass 1's scan and the CAS will either:
     *   (a) have already CAS'd rwlock from cur to cur+1 -- our CAS then
     *       fails (cur mismatched), recovery yields and a future
     *       cycle retries; or
     *   (b) be still in the subcount-bump phase -- our CAS sees the
     *       stale cur and resets to 0; the new reader's subsequent CAS
     *       rwlock(0 -> 1) succeeds cleanly.
     * Only after the CAS resolves do we wipe the deferred dead slots,
     * keeping that work outside the race-sensitive window. */
    /* A live reader with no slot (table was full) is invisible to the scan
     * above but still holds a +1 in the lock word; never force-reset under it. */
    if (__atomic_load_n(&hdr->slotless_readers, __ATOMIC_RELAXED) > 0)
        any_live_reader = 1;
    if (found_dead_reader && !any_live_reader) {
        /* ACQUIRE: a late reader's subcount++ (before its rwlock CAS) is then visible below. */
        uint32_t cur = __atomic_load_n(&hdr->rwlock, __ATOMIC_ACQUIRE);
        int drain_ok = 1;   /* keep dead slots if the reset doesn't fire */
        if (cur > 0 && cur < KD_RWLOCK_WRITER_BIT) {
            /* Re-scan for a live reader (fail-safe: only suppresses a reset). */
            int live_now = __atomic_load_n(&hdr->slotless_readers, __ATOMIC_RELAXED) > 0;
            for (uint32_t i = 0; !live_now && i < KD_READER_SLOTS; i++) {
                uint32_t p = __atomic_load_n(&h->reader_slots[i].pid, __ATOMIC_ACQUIRE);
                if (p && kd_pid_alive(p) &&
                    __atomic_load_n(&h->reader_slots[i].subcount, __ATOMIC_RELAXED) > 0)
                    live_now = 1;
            }
            if (live_now) {
                drain_ok = 0;
            } else if (__atomic_compare_exchange_n(&hdr->rwlock, &cur, 0,
                    0, __ATOMIC_RELEASE, __ATOMIC_RELAXED)) {
                if (__atomic_load_n(&hdr->rwlock_waiters, __ATOMIC_RELAXED) > 0)
                    syscall(SYS_futex, &hdr->rwlock, FUTEX_WAKE, INT_MAX, NULL, NULL, 0);
            } else {
                drain_ok = 0;   /* rwlock changed under us -- shares may still be live */
            }
        }
        if (drain_ok) {
            for (uint32_t i = 0; i < KD_READER_SLOTS; i++) {
                uint32_t p = __atomic_load_n(&h->reader_slots[i].pid, __ATOMIC_ACQUIRE);
                if (p == 0 || kd_pid_alive(p)) continue;
                kd_drain_dead_slot(h, i, p);
            }
        }
    }
}

/* Inspect the lock word after a futex-wait timeout.  If a dead writer
 * holds it, force-recover the lock.  Otherwise drain dead readers' shares
 * of the rwlock/waiter counters.  Called from rdlock and wrlock ETIMEDOUT
 * branches -- identical recovery logic in both. */
static inline void kd_recover_after_timeout(KdHandle *h) {
    KdHeader *hdr = h->hdr;
    uint32_t val = __atomic_load_n(&hdr->rwlock, __ATOMIC_RELAXED);
    if (val >= KD_RWLOCK_WRITER_BIT) {
        uint32_t pid = val & KD_RWLOCK_PID_MASK;
        if (!kd_pid_alive(pid))
            kd_recover_stale_lock(h, val);
    } else {
        kd_recover_dead_readers(h);
    }
}

/* Park/unpark helpers: bump the global waiter counters together with this
 * process's mirrored slot counters so a wrlock-timeout recovery scan can
 * attribute and reverse a dead PID's contribution.  Kept paired to make
 * accidental drift between global and per-slot counts impossible. */
static inline void kd_park_reader(KdHandle *h) {
    if (h->my_slot_idx != UINT32_MAX)
        __atomic_add_fetch(&h->reader_slots[h->my_slot_idx].waiters_parked, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&h->hdr->rwlock_waiters, 1, __ATOMIC_RELAXED);
}
static inline void kd_unpark_reader(KdHandle *h) {
    __atomic_sub_fetch(&h->hdr->rwlock_waiters, 1, __ATOMIC_RELAXED);
    if (h->my_slot_idx != UINT32_MAX)
        __atomic_sub_fetch(&h->reader_slots[h->my_slot_idx].waiters_parked, 1, __ATOMIC_RELAXED);
}
static inline void kd_park_writer(KdHandle *h) {
    if (h->my_slot_idx != UINT32_MAX) {
        __atomic_add_fetch(&h->reader_slots[h->my_slot_idx].waiters_parked, 1, __ATOMIC_RELAXED);
        __atomic_add_fetch(&h->reader_slots[h->my_slot_idx].writers_parked, 1, __ATOMIC_RELAXED);
    }
    __atomic_add_fetch(&h->hdr->rwlock_waiters, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&h->hdr->rwlock_writers_waiting, 1, __ATOMIC_RELAXED);
}
static inline void kd_unpark_writer(KdHandle *h) {
    __atomic_sub_fetch(&h->hdr->rwlock_waiters, 1, __ATOMIC_RELAXED);
    __atomic_sub_fetch(&h->hdr->rwlock_writers_waiting, 1, __ATOMIC_RELAXED);
    if (h->my_slot_idx != UINT32_MAX) {
        __atomic_sub_fetch(&h->reader_slots[h->my_slot_idx].waiters_parked, 1, __ATOMIC_RELAXED);
        __atomic_sub_fetch(&h->reader_slots[h->my_slot_idx].writers_parked, 1, __ATOMIC_RELAXED);
    }
}

/* Reader accounting: a reader mirrors its +1 in the lock word so dead-reader
 * recovery can see it. A slotted reader uses its slot subcount; a reader that
 * could not claim a slot (table full) uses the global hdr->slotless_readers,
 * so recovery's force-reset never fires out from under it. leave() peels
 * slotless first so a later slot claim cannot misattribute the decrement. */
static inline void kd_reader_enter(KdHandle *h) {
    if (h->my_slot_idx != UINT32_MAX) {
        __atomic_add_fetch(&h->reader_slots[h->my_slot_idx].subcount, 1, __ATOMIC_RELAXED);
    } else {
        __atomic_add_fetch(&h->hdr->slotless_readers, 1, __ATOMIC_RELAXED);
        h->slotless_held++;
    }
}
static inline void kd_reader_leave(KdHandle *h) {
    if (h->slotless_held > 0) {
        h->slotless_held--;
        __atomic_sub_fetch(&h->hdr->slotless_readers, 1, __ATOMIC_RELAXED);
    } else if (h->my_slot_idx != UINT32_MAX) {
        __atomic_sub_fetch(&h->reader_slots[h->my_slot_idx].subcount, 1, __ATOMIC_RELAXED);
    }
}

static inline void kd_rwlock_rdlock(KdHandle *h) {
    kd_claim_reader_slot(h);
    KdHeader *hdr = h->hdr;
    uint32_t *lock = &hdr->rwlock;
    uint32_t *writers_waiting = &hdr->rwlock_writers_waiting;
    /* Claim subcount BEFORE bumping the shared rwlock counter.  This way
     * a concurrent writer-side recovery scan that sees our PID alive with
     * subcount > 0 will (correctly) defer force-reset, even while we are
     * still spinning trying to win the rwlock CAS.  Without this, a reader
     * killed between rwlock CAS-success and subcount++ would let recovery
     * force-reset rwlock to 0 underneath us, causing a UINT32_MAX wrap on
     * our eventual rdunlock dec. */
    kd_reader_enter(h);
    for (int spin = 0; ; spin++) {
        uint32_t cur = __atomic_load_n(lock, __ATOMIC_RELAXED);
        /* Write-preferring: when lock is free (cur==0) and writers are
         * waiting, yield to let the writer acquire. When readers are
         * already active (cur>=1), new readers may join freely. */
        if (cur > 0 && cur < KD_RWLOCK_WRITER_BIT) {
            if (__atomic_compare_exchange_n(lock, &cur, cur + 1,
                    1, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
                return;
        } else if (cur == 0 && !__atomic_load_n(writers_waiting, __ATOMIC_RELAXED)) {
            if (__atomic_compare_exchange_n(lock, &cur, 1,
                    1, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
                return;
        }
        if (__builtin_expect(spin < KD_RWLOCK_SPIN_LIMIT, 1)) {
            kd_rwlock_spin_pause();
            continue;
        }
        kd_park_reader(h);
        cur = __atomic_load_n(lock, __ATOMIC_RELAXED);
        /* Sleep when write-locked OR when yielding to waiting writers */
        if (cur >= KD_RWLOCK_WRITER_BIT || cur == 0) {
            long rc = syscall(SYS_futex, lock, FUTEX_WAIT, cur,
                              &kd_lock_timeout, NULL, 0);
            if (rc == -1 && errno == ETIMEDOUT) {
                kd_unpark_reader(h);
                kd_recover_after_timeout(h);
                spin = 0;
                continue;
            }
        }
        kd_unpark_reader(h);
        spin = 0;
    }
}

static inline void kd_rwlock_rdunlock(KdHandle *h) {
    KdHeader *hdr = h->hdr;
    /* Release the shared counter BEFORE dropping our subcount so that
     * "any live PID with subcount > 0" is a reliable in-flight indicator
     * for the writer-side recovery scan.  Inverting these would create a
     * window where we still own a unit of rwlock but our slot subcount is
     * 0, letting recovery force-reset rwlock underneath us. */
    uint32_t after = __atomic_sub_fetch(&hdr->rwlock, 1, __ATOMIC_RELEASE);
    kd_reader_leave(h);
    if (after == 0 && __atomic_load_n(&hdr->rwlock_waiters, __ATOMIC_RELAXED) > 0)
        syscall(SYS_futex, &hdr->rwlock, FUTEX_WAKE, INT_MAX, NULL, NULL, 0);
}

static inline void kd_rwlock_wrlock(KdHandle *h) {
    kd_claim_reader_slot(h);  /* refresh cached_pid across fork */
    KdHeader *hdr = h->hdr;
    uint32_t *lock = &hdr->rwlock;
    /* Encode PID in the rwlock word itself (0x80000000 | pid) to eliminate
     * any crash window between acquiring the lock and storing the owner. */
    uint32_t mypid = KD_RWLOCK_WR(h->cached_pid);
    for (int spin = 0; ; spin++) {
        uint32_t expected = 0;
        if (__atomic_compare_exchange_n(lock, &expected, mypid,
                1, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED))
            return;
        if (__builtin_expect(spin < KD_RWLOCK_SPIN_LIMIT, 1)) {
            kd_rwlock_spin_pause();
            continue;
        }
        kd_park_writer(h);
        uint32_t cur = __atomic_load_n(lock, __ATOMIC_RELAXED);
        if (cur != 0) {
            long rc = syscall(SYS_futex, lock, FUTEX_WAIT, cur,
                              &kd_lock_timeout, NULL, 0);
            if (rc == -1 && errno == ETIMEDOUT) {
                kd_unpark_writer(h);
                kd_recover_after_timeout(h);
                spin = 0;
                continue;
            }
        }
        kd_unpark_writer(h);
        spin = 0;
    }
}

static inline void kd_rwlock_wrunlock(KdHandle *h) {
    KdHeader *hdr = h->hdr;
    __atomic_store_n(&hdr->rwlock, 0, __ATOMIC_RELEASE);
    if (__atomic_load_n(&hdr->rwlock_waiters, __ATOMIC_RELAXED) > 0)
        syscall(SYS_futex, &hdr->rwlock, FUTEX_WAKE, INT_MAX, NULL, NULL, 0);
}

/* ================================================================
 * Layout math + create / open / destroy
 *
 * Layout: Header -> reader_slots[1024] -> nodes[capacity] -> build_idx[capacity]
 * ================================================================ */

/* Single source of truth for the mmap region layout offsets.
 * Layout: Header -> reader_slots[1024] -> nodes[capacity] -> build_idx[capacity] */
typedef struct { uint64_t reader_slots, nodes, idx, total; } KdLayout;

static inline uint64_t kd_stride(uint32_t dims) {
    return (uint64_t)dims * sizeof(double) + 2 * sizeof(uint32_t) + sizeof(uint64_t);  /* coords + left/right + payload */
}

static inline KdLayout kd_layout_for(uint32_t dims, uint32_t capacity) {
    KdLayout L;
    uint64_t stride = kd_stride(dims);
    L.reader_slots = sizeof(KdHeader);
    L.nodes        = L.reader_slots + (uint64_t)KD_READER_SLOTS * sizeof(KdReaderSlot);
    L.nodes        = (L.nodes + 7) & ~(uint64_t)7;
    L.idx          = L.nodes + (uint64_t)capacity * stride;
    L.idx          = (L.idx + 7) & ~(uint64_t)7;
    L.total        = L.idx + (uint64_t)capacity * sizeof(uint32_t);
    return L;
}

static inline uint64_t kd_total_size(uint32_t dims, uint32_t capacity) {
    return kd_layout_for(dims, capacity).total;
}

static inline void kd_init_header(void *base, uint32_t dims, uint32_t capacity, uint64_t total) {
    KdLayout L = kd_layout_for(dims, capacity);
    KdHeader *hdr = (KdHeader *)base;
    memset(base, 0, (size_t)L.nodes);   /* header + reader slots; node data is written on add */
    hdr->magic            = KD_MAGIC;
    hdr->version          = KD_VERSION;
    hdr->dims             = dims;
    hdr->capacity         = capacity;
    hdr->count            = 0;
    hdr->root             = KD_NIL;
    hdr->dirty            = 0;
    hdr->node_stride      = kd_stride(dims);
    hdr->nodes_off        = L.nodes;
    hdr->idx_off          = L.idx;
    hdr->total_size       = total;
    hdr->reader_slots_off = L.reader_slots;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

/* ---- node accessors ---- */
static inline char    *kd_node(KdHandle *h, uint64_t i) { return (char *)h->base + h->nodes_off + i * h->node_stride; }
static inline double  *kd_coords(KdHandle *h, uint64_t i) { return (double *)kd_node(h, i); }
static inline uint64_t *kd_payload(KdHandle *h, uint64_t i) { return (uint64_t *)(kd_node(h, i) + (uint64_t)h->dims * sizeof(double)); }
static inline uint32_t *kd_left(KdHandle *h, uint64_t i)  { return (uint32_t *)(kd_node(h, i) + (uint64_t)h->dims * sizeof(double) + sizeof(uint64_t)); }
static inline uint32_t *kd_right(KdHandle *h, uint64_t i) { return (uint32_t *)(kd_node(h, i) + (uint64_t)h->dims * sizeof(double) + sizeof(uint64_t) + sizeof(uint32_t)); }
static inline uint32_t *kd_idx(KdHandle *h) { return (uint32_t *)((char *)h->base + h->idx_off); }

/* Layer B trusted bound: number of nodes guaranteed within the real mapping.
 * Equals capacity for a valid tree; every child index read from shared memory is
 * checked against it so a corrupt link can never drive an access out of bounds. */
static inline uint64_t kd_nodes_max(KdHandle *h) {
    if (h->nodes_off >= h->mmap_size || h->node_stride == 0) return 0;
    return (h->mmap_size - h->nodes_off) / h->node_stride;
}
#define KD_NODE_OK(h, i) ((uint32_t)(i) != KD_NIL && (uint64_t)(i) < (uint64_t)(h)->capacity)

static inline KdHandle *kd_setup(void *base, size_t map_size,
                                 const char *path, int backing_fd) {
    KdHeader *hdr = (KdHeader *)base;
    KdHandle *h = (KdHandle *)calloc(1, sizeof(KdHandle));
    if (!h) {
        munmap(base, map_size);
        if (backing_fd >= 0) close(backing_fd);
        return NULL;
    }
    h->hdr          = hdr;
    h->base         = base;
    h->reader_slots = (KdReaderSlot *)((uint8_t *)base + hdr->reader_slots_off);
    h->nodes_off    = hdr->nodes_off;   /* single validated read of each geometry field */
    h->idx_off      = hdr->idx_off;
    h->node_stride  = hdr->node_stride;
    h->dims         = hdr->dims;
    h->capacity     = hdr->capacity;
    h->mmap_size    = map_size;
    /* Layer B: clamp the cached capacity to the number of nodes that actually fit */
    {
        uint64_t fit = kd_nodes_max(h);
        if ((uint64_t)h->capacity > fit) h->capacity = (uint32_t)fit;
    }
    h->path         = path ? strdup(path) : NULL;
    h->backing_fd   = backing_fd;
    h->my_slot_idx  = UINT32_MAX;
    return h;
}

/* Validate a mapped header (shared by kd_create reopen and kd_open_fd). */
static inline int kd_validate_header(const KdHeader *hdr, uint64_t file_size) {
    if (hdr->magic != KD_MAGIC) return 0;
    if (hdr->version != KD_VERSION) return 0;
    if (hdr->dims < KD_MIN_DIMS || hdr->dims > KD_MAX_DIMS) return 0;
    if (hdr->capacity < KD_MIN_CAP || hdr->capacity > KD_MAX_CAP) return 0;
    if (hdr->count > hdr->capacity) return 0;
    if (hdr->node_stride != kd_stride(hdr->dims)) return 0;
    if (hdr->total_size != file_size) return 0;
    if (hdr->total_size != kd_total_size(hdr->dims, hdr->capacity)) return 0;
    KdLayout L = kd_layout_for(hdr->dims, hdr->capacity);
    if (hdr->reader_slots_off != L.reader_slots) return 0;
    if (hdr->nodes_off != L.nodes) return 0;
    if (hdr->idx_off != L.idx) return 0;
    return 1;
}

/* validate the requested dims + capacity */
static int kd_validate_args(uint64_t dims, uint64_t capacity, char *errbuf) {
    if (errbuf) errbuf[0] = '\0';
    if (dims < KD_MIN_DIMS || dims > KD_MAX_DIMS) { KD_ERR("dims must be between 1 and 16"); return 0; }
    if (capacity < KD_MIN_CAP || capacity > KD_MAX_CAP) { KD_ERR("capacity must be between 1 and 2^24"); return 0; }
    return 1;
}

/* Securely obtain a fd for a path-backed segment: create it exclusively
 * (O_CREAT|O_EXCL|O_NOFOLLOW at `mode`, default 0600 = owner-only), or, if it
 * already exists, attach to it (O_RDWR|O_NOFOLLOW, no O_CREAT). O_EXCL blocks a
 * pre-seeded or hard-linked file and O_NOFOLLOW a symlink swap, so a local
 * attacker can no longer redirect or poison the backing store through the path.
 * Cross-user sharing is opt-in via a wider `mode` (e.g. 0660); the caller still
 * validates the file's contents via kd_validate_header. */
static int kd_secure_open(const char *path, mode_t mode, char *errbuf) {
    for (int attempt = 0; attempt < 100; attempt++) {
        int fd = open(path, O_RDWR|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC, mode);
        if (fd >= 0) { (void)fchmod(fd, mode); return fd; }   /* exact mode: umask narrowed the O_EXCL create */
        if (errno != EEXIST) { KD_ERR("create %s: %s", path, strerror(errno)); return -1; }
        fd = open(path, O_RDWR|O_NOFOLLOW|O_CLOEXEC);
        if (fd >= 0) return fd;
        if (errno == ENOENT) continue;   /* creator unlinked between our two opens; retry */
        KD_ERR("open %s: %s", path, strerror(errno));  /* ELOOP => symlink rejected */
        return -1;
    }
    KD_ERR("open %s: create/attach kept racing", path);
    return -1;
}

static KdHandle *kd_create(const char *path, uint64_t dims, uint64_t capacity, mode_t mode, char *errbuf) {
    if (!kd_validate_args(dims, capacity, errbuf)) return NULL;

    uint64_t total = kd_total_size((uint32_t)dims, (uint32_t)capacity);
    int anonymous = (path == NULL);
    int fd = -1;
    size_t map_size;
    void *base;

    if (anonymous) {
        map_size = (size_t)total;
        base = mmap(NULL, map_size, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);
        if (base == MAP_FAILED) { KD_ERR("mmap: %s", strerror(errno)); return NULL; }
    } else {
        fd = kd_secure_open(path, mode, errbuf);
        if (fd < 0) return NULL;
        if (flock(fd, LOCK_EX) < 0) { KD_ERR("flock: %s", strerror(errno)); close(fd); return NULL; }
        struct stat st;
        if (fstat(fd, &st) < 0) { KD_ERR("fstat: %s", strerror(errno)); flock(fd, LOCK_UN); close(fd); return NULL; }
        int is_new = (st.st_size == 0);
        if (!is_new && (uint64_t)st.st_size < sizeof(KdHeader)) {
            KD_ERR("%s: file too small (%lld)", path, (long long)st.st_size);
            flock(fd, LOCK_UN); close(fd); return NULL;
        }
        if (is_new && ftruncate(fd, (off_t)total) < 0) {
            KD_ERR("ftruncate: %s", strerror(errno)); flock(fd, LOCK_UN); close(fd); return NULL;
        }
        map_size = is_new ? (size_t)total : (size_t)st.st_size;
        base = mmap(NULL, map_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
        if (base == MAP_FAILED) { KD_ERR("mmap: %s", strerror(errno)); flock(fd, LOCK_UN); close(fd); return NULL; }
        if (!is_new) {
            if (!kd_validate_header((KdHeader *)base, (uint64_t)st.st_size)) {
                KD_ERR("invalid k-d tree file"); munmap(base, map_size); flock(fd, LOCK_UN); close(fd); return NULL;
            }
            flock(fd, LOCK_UN); close(fd);
            return kd_setup(base, map_size, path, -1);
        }
    }
    kd_init_header(base, (uint32_t)dims, (uint32_t)capacity, total);
    if (fd >= 0) { flock(fd, LOCK_UN); close(fd); }
    return kd_setup(base, map_size, path, -1);
}

static KdHandle *kd_create_memfd(const char *name, uint64_t dims, uint64_t capacity, char *errbuf) {
    if (!kd_validate_args(dims, capacity, errbuf)) return NULL;

    uint64_t total = kd_total_size((uint32_t)dims, (uint32_t)capacity);
    int fd = memfd_create(name ? name : "kdtree", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (fd < 0) { KD_ERR("memfd_create: %s", strerror(errno)); return NULL; }
    if (ftruncate(fd, (off_t)total) < 0) {
        KD_ERR("ftruncate: %s", strerror(errno)); close(fd); return NULL;
    }
    (void)fcntl(fd, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW);
    void *base = mmap(NULL, (size_t)total, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { KD_ERR("mmap: %s", strerror(errno)); close(fd); return NULL; }
    kd_init_header(base, (uint32_t)dims, (uint32_t)capacity, total);
    return kd_setup(base, (size_t)total, NULL, fd);
}

static KdHandle *kd_open_fd(int fd, char *errbuf) {
    if (errbuf) errbuf[0] = '\0';
    struct stat st;
    if (fstat(fd, &st) < 0) { KD_ERR("fstat: %s", strerror(errno)); return NULL; }
    if ((uint64_t)st.st_size < sizeof(KdHeader)) { KD_ERR("too small"); return NULL; }
    size_t ms = (size_t)st.st_size;
    void *base = mmap(NULL, ms, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { KD_ERR("mmap: %s", strerror(errno)); return NULL; }
    if (!kd_validate_header((KdHeader *)base, (uint64_t)st.st_size)) {
        KD_ERR("invalid k-d tree table"); munmap(base, ms); return NULL;
    }
    int myfd = fcntl(fd, F_DUPFD_CLOEXEC, 0);
    if (myfd < 0) { KD_ERR("fcntl: %s", strerror(errno)); munmap(base, ms); return NULL; }
    return kd_setup(base, ms, NULL, myfd);
}

static void kd_destroy(KdHandle *h) {
    if (!h) return;
    /* Release our reader slot on clean teardown (else short-lived-reader churn
     * exhausts the slot table); skip if a lock is still held (subcount>0). */
    if (h->reader_slots && h->my_slot_idx != UINT32_MAX && h->cached_pid &&
        h->cached_fork_gen == __atomic_load_n(&kd_fork_gen, __ATOMIC_RELAXED) &&
        __atomic_load_n(&h->reader_slots[h->my_slot_idx].subcount, __ATOMIC_ACQUIRE) == 0) {
        uint32_t expected = h->cached_pid;
        __atomic_compare_exchange_n(&h->reader_slots[h->my_slot_idx].pid,
                &expected, 0, 0, __ATOMIC_RELEASE, __ATOMIC_RELAXED);
    }
    if (h->backing_fd >= 0) close(h->backing_fd);
    if (h->base) munmap(h->base, h->mmap_size);
    free(h->path);
    free(h);
}

static inline int kd_msync(KdHandle *h) {
    if (!h || !h->base) return 0;
    return msync(h->base, h->mmap_size, MS_SYNC);
}

/* ================================================================
 * k-d tree operations (callers hold the lock)
 *
 * Points are appended O(1) and marked dirty; a balanced tree is bulk-built by
 * median split on the first query after any insert, so query recursion is
 * O(log n) deep whatever the insertion order.  Nearest / k-nearest use a bounded
 * max-heap with plane-distance pruning; range and radius prune by the splitting
 * plane.  Every child index read from shared memory is bounds-checked.
 * ================================================================ */

/* squared Euclidean distance from node `node`'s coords to query point q */
static inline double kd_dist2_to(KdHandle *h, uint64_t node, const double *q) {
    double *c = kd_coords(h, node);
    double s = 0.0;
    for (uint32_t d = 0; d < h->dims; d++) { double diff = c[d] - q[d]; s += diff * diff; }
    return s;
}

/* append a point; returns its slot index, or -1 if the tree is full.  Marks the
 * tree dirty so the next query rebuilds a balanced tree.  (caller holds wrlock) */
static int64_t kd_add_locked(KdHandle *h, const double *coords, uint64_t payload) {
    uint64_t slot = h->hdr->count;
    if (slot >= h->capacity) return -1;                   /* full */
    double *c = kd_coords(h, slot);
    for (uint32_t d = 0; d < h->dims; d++) c[d] = coords[d];
    *kd_payload(h, slot) = payload;
    *kd_left(h, slot)    = KD_NIL;
    *kd_right(h, slot)   = KD_NIL;
    h->hdr->count = slot + 1;
    h->hdr->dirty = 1;
    return (int64_t)slot;
}

/* ---- balanced bulk build (median split on a cycling axis) ---- */
typedef struct { KdHandle *h; uint32_t axis; } KdCmpCtx;

static int kd_cmp_axis(const void *pa, const void *pb, void *arg) {
    KdCmpCtx *ctx = (KdCmpCtx *)arg;
    uint32_t a = *(const uint32_t *)pa, b = *(const uint32_t *)pb;
    double va = kd_coords(ctx->h, a)[ctx->axis];
    double vb = kd_coords(ctx->h, b)[ctx->axis];
    if (va < vb) return -1;
    if (va > vb) return  1;
    return (a < b) ? -1 : (a > b ? 1 : 0);                /* stable tiebreak */
}

static uint32_t kd_build_rec(KdHandle *h, uint32_t *idx, int64_t lo, int64_t hi, uint32_t depth) {
    if (lo > hi) return KD_NIL;
    KdCmpCtx ctx = { h, depth % h->dims };
    qsort_r(idx + lo, (size_t)(hi - lo + 1), sizeof(uint32_t), kd_cmp_axis, &ctx);
    int64_t mid = lo + (hi - lo) / 2;
    uint32_t node = idx[mid];
    *kd_left(h, node)  = kd_build_rec(h, idx, lo, mid - 1, depth + 1);
    *kd_right(h, node) = kd_build_rec(h, idx, mid + 1, hi, depth + 1);
    return node;
}

/* (re)build a balanced tree over all inserted points (caller holds wrlock) */
static void kd_build_locked(KdHandle *h) {
    uint64_t n = h->hdr->count;
    if (n > h->capacity) n = h->capacity;                 /* Layer B */
    if (n == 0) { h->hdr->root = KD_NIL; h->hdr->dirty = 0; return; }
    uint32_t *idx = kd_idx(h);                             /* scratch region inside the mapping */
    for (uint64_t i = 0; i < n; i++) idx[i] = (uint32_t)i;
    h->hdr->root = kd_build_rec(h, idx, 0, (int64_t)n - 1, 0);
    h->hdr->dirty = 0;
}

/* ---- k-nearest search via a bounded max-heap of the m best (farthest at 0) ---- */
typedef struct { uint64_t id; double dist2; } KdRes;

static void kd_heap_offer(KdRes *heap, uint64_t *cnt, uint64_t m, uint64_t id, double d2) {
    if (*cnt < m) {                                        /* grow the heap */
        uint64_t i = (*cnt)++;
        heap[i].id = id; heap[i].dist2 = d2;
        while (i > 0) { uint64_t p = (i - 1) / 2;
            if (heap[p].dist2 >= heap[i].dist2) break;
            KdRes t = heap[p]; heap[p] = heap[i]; heap[i] = t; i = p; }
    } else if (m > 0 && d2 < heap[0].dist2) {              /* replace the current farthest */
        heap[0].id = id; heap[0].dist2 = d2;
        uint64_t i = 0;
        for (;;) { uint64_t l = 2*i+1, r = 2*i+2, big = i;
            if (l < m && heap[l].dist2 > heap[big].dist2) big = l;
            if (r < m && heap[r].dist2 > heap[big].dist2) big = r;
            if (big == i) break;
            KdRes t = heap[big]; heap[big] = heap[i]; heap[i] = t; i = big; }
    }
}

static void kd_knn_rec(KdHandle *h, uint32_t node, const double *q, uint32_t depth,
                       KdRes *heap, uint64_t *cnt, uint64_t m) {
    if (!KD_NODE_OK(h, node) || depth > KD_MAX_DEPTH) return;   /* Layer B: bound recursion depth */
    kd_heap_offer(heap, cnt, m, *kd_payload(h, node), kd_dist2_to(h, node, q));
    uint32_t axis = depth % h->dims;
    double diff = q[axis] - kd_coords(h, node)[axis];
    uint32_t near = (diff < 0) ? *kd_left(h, node)  : *kd_right(h, node);
    uint32_t far  = (diff < 0) ? *kd_right(h, node) : *kd_left(h, node);
    kd_knn_rec(h, near, q, depth + 1, heap, cnt, m);
    if (*cnt < m || diff * diff < heap[0].dist2)          /* the far side may hold a closer point */
        kd_knn_rec(h, far, q, depth + 1, heap, cnt, m);
}

/* fill up to m nearest points to q into heap[]; returns the number found.
 * heap must have room for m entries.  (caller holds a query lock) */
static uint64_t kd_knn_locked(KdHandle *h, const double *q, uint64_t m, KdRes *heap) {
    uint64_t cnt = 0;
    if (m == 0) return 0;
    kd_knn_rec(h, h->hdr->root, q, 0, heap, &cnt, m);
    return cnt;
}

/* ---- axis-aligned box (range) search ---- */
static void kd_range_rec(KdHandle *h, uint32_t node, const double *lo, const double *hi,
                         uint32_t depth, uint64_t *out, uint64_t *cnt, uint64_t cap) {
    if (!KD_NODE_OK(h, node) || depth > KD_MAX_DEPTH) return;   /* Layer B: bound recursion depth */
    double *c = kd_coords(h, node);
    int inside = 1;
    for (uint32_t d = 0; d < h->dims; d++) if (c[d] < lo[d] || c[d] > hi[d]) { inside = 0; break; }
    if (inside && *cnt < cap) out[(*cnt)++] = *kd_payload(h, node);
    uint32_t axis = depth % h->dims;
    if (lo[axis] <= c[axis]) kd_range_rec(h, *kd_left(h, node),  lo, hi, depth + 1, out, cnt, cap);
    if (hi[axis] >= c[axis]) kd_range_rec(h, *kd_right(h, node), lo, hi, depth + 1, out, cnt, cap);
}

static uint64_t kd_range_locked(KdHandle *h, const double *lo, const double *hi,
                                uint64_t *out, uint64_t cap) {
    uint64_t cnt = 0;
    kd_range_rec(h, h->hdr->root, lo, hi, 0, out, &cnt, cap);
    return cnt;
}

/* ---- radius (ball) search ---- */
static void kd_radius_rec(KdHandle *h, uint32_t node, const double *q, double r, double r2,
                          uint32_t depth, KdRes *out, uint64_t *cnt, uint64_t cap) {
    if (!KD_NODE_OK(h, node) || depth > KD_MAX_DEPTH) return;   /* Layer B: bound recursion depth */
    double d2 = kd_dist2_to(h, node, q);
    if (d2 <= r2 && *cnt < cap) { out[*cnt].id = *kd_payload(h, node); out[*cnt].dist2 = d2; (*cnt)++; }
    uint32_t axis = depth % h->dims;
    double c = kd_coords(h, node)[axis];
    if (q[axis] - r <= c) kd_radius_rec(h, *kd_left(h, node),  q, r, r2, depth + 1, out, cnt, cap);
    if (q[axis] + r >= c) kd_radius_rec(h, *kd_right(h, node), q, r, r2, depth + 1, out, cnt, cap);
}

static uint64_t kd_radius_locked(KdHandle *h, const double *q, double r,
                                 KdRes *out, uint64_t cap) {
    uint64_t cnt = 0;
    kd_radius_rec(h, h->hdr->root, q, r, r * r, 0, out, &cnt, cap);
    return cnt;
}

/* reset to an empty tree (caller holds the write lock) */
static inline void kd_clear_locked(KdHandle *h) {
    h->hdr->count = 0;
    h->hdr->root  = KD_NIL;
    h->hdr->dirty = 0;
}

#endif /* KD_H */
