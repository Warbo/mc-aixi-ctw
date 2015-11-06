#ifndef __LOG_HPP__
#define __LOG_HPP__

#include <boost/logging/format_fwd.hpp>

BOOST_LOG_FORMAT_MSG( optimize::cache_string_one_str<> )

#ifndef BOOST_LOG_COMPILE_FAST
#include <boost/logging/format.hpp>
#include <boost/logging/writer/ts_write.hpp>
#endif

typedef boost::logging::logger_format_write< > log_type;

BOOST_DECLARE_LOG_FILTER(g_l_filter, boost::logging::level::holder)
BOOST_DECLARE_LOG(g_l, log_type)

#define LDBG_ BOOST_LOG_USE_LOG_IF_LEVEL(g_l(), g_l_filter(), debug ) << "[DEBUG] "
#define LERR_ BOOST_LOG_USE_LOG_IF_LEVEL(g_l(), g_l_filter(), error ) << "[ERROR] "
#define LINFO_ BOOST_LOG_USE_LOG_IF_LEVEL(g_l(), g_l_filter(), info ) << "[INFO]  "

// initialise the logging mechanism
extern void initLogs();

#endif  // __LOG_HPP__

