#ifndef LOGGING_H_
#define LOGGING_H_

#include <iomanip>
#include <iostream>
#include <llis/utils/time.hh>

#define N_FIXED_CTR 0
#define N_CUSTOM_CTR 4

/*****************************************************************
 *********************** LOGGING MACROS   ************************
 *****************************************************************/
#define LOG_FD stderr

//#define LOG_DEBUG

/* For coloring log output  */
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#define ANSI_COLOR_PURPLE   "\x1b[35m"

/* General logging function which can be filled in with arguments, color, etc. */
#define log_at_level(lvl_label, color, fd, fmt, ...)\
        fprintf(fd, "" color "%07.03f:%s:%d:%s(): " lvl_label ": " fmt ANSI_COLOR_RESET "\n", \
                ((std::chrono::duration<double>)(hr_clock::now() - system_start_time)).count(), \
                __FILE__, __LINE__, __func__, ##__VA_ARGS__)

/* Debug statements are replaced with nothing if LOG_DEBUG is false  */
#ifdef LOG_DEBUG
#define log_debug(fmt, ...)\
    log_at_level("DEBUG", ANSI_COLOR_RESET, LOG_FD, fmt, ##__VA_ARGS__)
#else
#define log_debug(...)
#endif

#define log_info(fmt, ...)\
    log_at_level("INFO", ANSI_COLOR_GREEN, LOG_FD, fmt, ##__VA_ARGS__)
#define log_error(fmt, ...)\
    log_at_level("ERROR", ANSI_COLOR_RED, LOG_FD, fmt, ##__VA_ARGS__)
#define log_warn(fmt, ...)\
    log_at_level("WARN", ANSI_COLOR_YELLOW, LOG_FD, fmt, ##__VA_ARGS__)

#ifdef PRINT_REQUEST_ERRORS
#define print_request_error(fmt, ...)\
    log_warn(fmt, ##__VA_ARGS__);
#else
#define print_request_error(...)
#endif

/**
 * Simple macro to replace perror with out log format
 */
#define log_perror(fmt, ...) \
    log_error(fmt ": %s", ##__VA_ARGS__, strerror(errno))

/**
 * Same as above, but to be used only for request-based errors
 */
#define perror_request(fmt, ...) \
    print_request_error(fmt ": %s", ##__VA_ARGS__, strerror(errno))

/* Some c++ macro */
#define LOG(lbl, color, ...)\
    std::cerr << color \
              << std::fixed << std::setw(7) << std::setprecision(3) << std::setfill('0') \
              << std::chrono::duration_cast<std::chrono::nanoseconds>(hr_clock::now() - system_start_time).count() \
              << ":" __FILE__ ":" << __LINE__ << ":" << __func__ << ":" \
              << lbl << ": " << __VA_ARGS__ \
              << ANSI_COLOR_RESET << "\n";

#ifdef LOG_DEBUG
#define LLIS_DEBUG(...) \
    LOG("DEBUG", ANSI_COLOR_RESET, __VA_ARGS__)
#else
#define LLIS_DEBUG(...)
#endif

#define LLIS_INFO(...) \
    LOG("INFO", ANSI_COLOR_GREEN, __VA_ARGS__)

#define LLIS_ERROR(...) \
    LOG("ERROR", ANSI_COLOR_RED, __VA_ARGS__)

#define LLIS_WARN(...) \
    LOG("WARN", ANSI_COLOR_YELLOW, __VA_ARGS__)

#endif //LOGGING_H_
