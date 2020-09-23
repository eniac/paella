#pragma once

#include <cstddef>

namespace llis {
namespace client {

/*
 * ptr: the pointer to the memory in the local address space
 * id: identifier that identify the instance of mmap that the piece of memory belongs to
 * offset: one mmap can involve multiple MemoryEntry, and the offset denotes the part of the mmap. Offset is in bytes
 * (id, offset) <=> ptr
 */
struct IoShmEntry {
    void* ptr;
    int id;
    size_t offset;
};

}
}

