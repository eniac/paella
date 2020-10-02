#include "shm_channel_impl.h"

namespace llis {
namespace ipc {

template class ShmChannelBase<true>;
#ifndef __CUDA_ARCH__
template class ShmChannelBase<false>;
#endif

}
}
