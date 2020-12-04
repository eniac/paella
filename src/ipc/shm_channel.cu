#include "shm_channel_impl.h"

namespace llis {
namespace ipc {

template <>
CUDA_HOSTDEV void* ShmChannelBase<false>::my_memcpy(void* dest, const void* src, size_t count) {
    return memcpy(dest, src, count);
}

template <>
CUDA_HOSTDEV void* ShmChannelBase<true>::my_memcpy(void* dest_, const void* src_, size_t count) {
    volatile char* dest = reinterpret_cast<volatile char*>(dest_);
    volatile const char* src = reinterpret_cast<volatile const char*>(src_);

    for (size_t i = 0; i < count; ++i) {
        dest[i] = src[i];
    }

    return dest_;
}

template class ShmChannelBase<true>;
#ifndef __CUDA_ARCH__
template class ShmChannelBase<false>;
#endif

}
}
