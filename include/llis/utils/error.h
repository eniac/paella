#include <boost/assert/source_location.hpp>
#include <boost/system/system_error.hpp>
#include <spdlog/common.h>
#include <spdlog/spdlog.h>
#include <cuda_runtime.h>

namespace llis {
namespace utils {

inline void error_throw_posix(int ret, int failed_ret = -1, const char* prefix = nullptr, boost::source_location location = BOOST_CURRENT_LOCATION) {
    if (ret != failed_ret) {
        return;
    }

    boost::system::error_code ec;
    ec.assign(errno, boost::system::system_category(), &location);

    if (prefix) {
	throw boost::system::system_error(ec, prefix);
    } else {
	throw boost::system::system_error(ec);
    }
}

class error_category_cuda_impl: public boost::system::error_category {
  public:
    error_category_cuda_impl() = default;

    const char* name() const noexcept {
	return "cuda";
    }

    std::string message(int ev) const {
	return cudaGetErrorString((cudaError_t)ev);
    }

    char const* message(int ev, char* buffer, std::size_t len) const noexcept {
	strncpy(buffer, cudaGetErrorString((cudaError_t)ev), len);
	return buffer;
    }
};

inline boost::system::error_category const& error_category_cuda() {
    static const error_category_cuda_impl instance;
    return instance;
}

inline void error_throw_cuda(cudaError_t err, const char* prefix = nullptr, boost::source_location location = BOOST_CURRENT_LOCATION) {
    if (err == cudaSuccess) {
	return;
    }

    boost::system::error_code ec;
    ec.assign(err, error_category_cuda(), &location);

    if (prefix) {
	throw boost::system::system_error(ec, prefix);
    } else {
	throw boost::system::system_error(ec);
    }
}

inline void error_throw_cuda_last(const char* prefix = nullptr, boost::source_location location = BOOST_CURRENT_LOCATION) {
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) {
	return;
    }

    boost::system::error_code ec;
    ec.assign(err, error_category_cuda(), &location);

    if (prefix) {
	throw boost::system::system_error(ec, prefix);
    } else {
	throw boost::system::system_error(ec);
    }
}

inline void error_log_posix(int ret, int failed_ret = -1, const char* prefix = nullptr, boost::source_location location = BOOST_CURRENT_LOCATION) {
    if (ret != failed_ret) {
        return;
    }

    spdlog::default_logger_raw()->log(spdlog::source_loc{location.file_name(), (int)location.line(), location.function_name()}, spdlog::level::err, "{}Error {}: {}", prefix ? prefix : "", errno, strerror(errno));
}

inline void error_log_cuda(cudaError_t err, const char* prefix = nullptr, boost::source_location location = BOOST_CURRENT_LOCATION) {
    if (err == cudaSuccess) {
        return;
    }

    spdlog::default_logger_raw()->log(spdlog::source_loc{location.file_name(), (int)location.line(), location.function_name()}, spdlog::level::err, "{}Error {}: {}", prefix ? prefix : "", err, cudaGetErrorString(err));
}

inline void error_log_cuda_last(const char* prefix = nullptr, boost::source_location location = BOOST_CURRENT_LOCATION) {
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) {
        return;
    }

    spdlog::default_logger_raw()->log(spdlog::source_loc{location.file_name(), (int)location.line(), location.function_name()}, spdlog::level::err, "{}Error {}: {}", prefix ? prefix : "", err, cudaGetErrorString(err));
}

inline void warn_log_posix(int ret, int failed_ret = -1, const char* prefix = nullptr, boost::source_location location = BOOST_CURRENT_LOCATION) {
    if (ret != failed_ret) {
        return;
    }

    spdlog::default_logger_raw()->log(spdlog::source_loc{location.file_name(), (int)location.line(), location.function_name()}, spdlog::level::warn, "{}Error {}: {}", prefix ? prefix : "", errno, strerror(errno));
}

inline void warn_log_cuda(cudaError_t err, const char* prefix = nullptr, boost::source_location location = BOOST_CURRENT_LOCATION) {
    if (err == cudaSuccess) {
        return;
    }

    spdlog::default_logger_raw()->log(spdlog::source_loc{location.file_name(), (int)location.line(), location.function_name()}, spdlog::level::warn, "{}Error {}: {}", prefix ? prefix : "", err, cudaGetErrorString(err));
}

inline void warn_log_cuda_last(const char* prefix = nullptr, boost::source_location location = BOOST_CURRENT_LOCATION) {
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) {
        return;
    }

    spdlog::default_logger_raw()->log(spdlog::source_loc{location.file_name(), (int)location.line(), location.function_name()}, spdlog::level::warn, "{}Error {}: {}", prefix ? prefix : "", err, cudaGetErrorString(err));
}

}
}

