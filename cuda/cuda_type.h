



#ifndef CUDA_TYPE_H
#define CUDA_TYPE_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t VertexId_CUDA;
const int CUDA_NUM_THREADS = 512;
const int CUDA_NUM_BLOCKS = 128;
const int CUDA_NUM_THREADS_SOFTMAX = 32;
const int CUDA_NUM_BLOCKS_SOFTMAX = 512;
}
#ifdef __cplusplus

#endif

#endif
