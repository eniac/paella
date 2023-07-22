#include <llis/server/scheduler.h>

namespace llis {
namespace server {

std::unordered_map<std::string, SchedulerFactory::RegisterFunc> SchedulerFactory::registered_schedulers_; 

bool SchedulerFactory::register_scheduler(std::string name, RegisterFunc func) {
    if (registered_schedulers_.find(name) == registered_schedulers_.end()) {
        registered_schedulers_.emplace(name, func);
        return true;
    } else {
        return false;
    }
}

std::unique_ptr<Scheduler> SchedulerFactory::create(std::string name, const po::variables_map& args) {
    auto it = registered_schedulers_.find(name);
    if (it == registered_schedulers_.end()) {
        return nullptr;
    } else {
        return it->second(args);
    }
}

}
}

