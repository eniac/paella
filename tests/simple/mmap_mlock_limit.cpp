#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cerrno>

int main(int argc, char** argv) {
    unsigned num = atoi(argv[1]);
    unsigned base_size = atoi(argv[2]);

    int shm_fd = shm_open("test_mmap_mlock_limit", O_RDWR | O_CREAT, 0600);

    if (shm_fd == -1) {
        printf("shm_open error: %d\n", errno);
    }

    size_t total_size = 0;
    for (int i = 0; i < num; ++i) {
        size_t this_size = base_size * (i + 1);
        total_size += this_size;

        int trunc_ret = ftruncate(shm_fd, total_size);
        if (trunc_ret == -1) {
            printf("ftruncate error: %d, i = %d, total_size: %lu, this_size: %lu\n", errno, i, total_size, this_size);
            break;
        }

        void* ptr = mmap(nullptr, this_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, total_size - this_size);
        if (ptr == MAP_FAILED) {
            printf("mmap error: %d, i = %d, total_size: %lu, this_size: %lu\n", errno, i, total_size, this_size);
            break;
        }

        int mlock_ret = mlock(ptr, this_size);
        if (mlock_ret == -1) {
            printf("mlock error: %d, i = %d, total_size: %lu, this_size: %lu\n", errno, i, total_size, this_size);
            break;
        }
    }

    shm_unlink("test_mmap_mlock_limit");
}

