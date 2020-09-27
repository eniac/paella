#include <llis/ipc/shm_channel.h>

#include <thread>

void reader(llis::ipc::ShmChannel* channel) {
    for (int i = 0; i < 10000; ++i) {
        int val;
        channel->read(&val, sizeof(val));
        if (val != i) {
            printf("Error! Expected: %d, Actual: %d\n", i, val);
            break;
        }
    }
}

void writer(llis::ipc::ShmChannel* channel) {
    for (int i = 0; i < 10000; ++i) {
        channel->write(i);
    }
}

int main() {
    llis::ipc::ShmChannel channel(64);

    std::thread reader_thr(reader, &channel);
    std::thread writer_thr(writer, &channel);

    reader_thr.join();
    writer_thr.join();
}

