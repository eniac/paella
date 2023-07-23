#pragma once

#include <string>
#include <filesystem>

namespace llis {
namespace utils {

namespace internal {

template <typename T>
std::filesystem::path path_concat_internal(T path_str) {
    return std::filesystem::path(path_str);
}

template <typename T, typename... Args>
std::filesystem::path path_concat_internal(T path_str, Args... paths_str) {
    std::filesystem::path res = path_concat_internal(paths_str...);
    res = std::filesystem::path(path_str) / res;
    return res;
}

}

template <typename... Args>
std::string path_concat(Args... paths_str) {
    return internal::path_concat_internal(paths_str...).string();
}

}
}
