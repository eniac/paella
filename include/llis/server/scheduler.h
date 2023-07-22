#pragma once

#include <llis/server/server.h>

#include <boost/program_options.hpp>

#include <functional>
#include <unordered_map>
#include <string>
#include <memory>

namespace llis {

namespace po = boost::program_options;

namespace server {

class Scheduler {
  public:
    virtual void set_server(Server* server) {
        server_ = server;
        profiler_ = server_->get_profiler();
    }

    virtual void try_handle_block_start_finish() = 0;
    virtual void handle_new_job(std::unique_ptr<job::Job> job) = 0;

  protected:
    Server* server_;
    Profiler* profiler_;
};

class SchedulerFactory {
  public:
    using RegisterFunc = std::function<std::unique_ptr<Scheduler>(const po::variables_map&)>;

    static bool register_scheduler(std::string name, RegisterFunc func);
    static std::unique_ptr<Scheduler> create(std::string name, const po::variables_map& args);

  private:
    static std::unordered_map<std::string, RegisterFunc> registered_schedulers_; 
};

#define LLIS_SCHEDULER_REGISTER(name, args) \
    static bool __scheduler_register_ = llis::server::SchedulerFactory::register_scheduler(name, args);

}
}
