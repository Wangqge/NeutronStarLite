

#ifndef ATOMIC_HPP
#define ATOMIC_HPP

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

template <class T> inline bool cas(T *ptr, T old_val, T new_val) {
  if (sizeof(T) == 8) {
    return __sync_bool_compare_and_swap((long *)ptr, *((long *)&old_val),
                                        *((long *)&new_val));
  } else if (sizeof(T) == 4) {
    return __sync_bool_compare_and_swap((int *)ptr, *((int *)&old_val),
                                        *((int *)&new_val));
  } else {
    assert(false);
  }
}

template <class T> inline bool write_min(T *ptr, T val) {
  volatile T curr_val;
  bool done = false;
  do {
    curr_val = *ptr;
  } while (curr_val > val && !(done = cas(ptr, curr_val, val)));
  return done;
}

template <class T> inline bool write_max(T *ptr, T val) {
  volatile T curr_val;
  bool done = false;
  do {
    curr_val = *ptr;
  } while (curr_val < val && !(done = cas(ptr, curr_val, val)));
  return done;
}

template <class T> inline void write_add(T *ptr, T val) {
  volatile T new_val, old_val;
  do {
    old_val = *ptr;
    new_val = old_val + val;
  } while (!cas(ptr, old_val, new_val));
}
template <class T> inline int get_and_reset(T *ptr) {
  volatile T new_val, old_val;
  do {
    old_val = *ptr;
    new_val = (T)0;
  } while (!cas(ptr, old_val, new_val));
}

#endif
