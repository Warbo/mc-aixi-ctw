#include "log.hpp"

#include <boost/logging/format.hpp>
#include <boost/logging/format/formatter/named_spacer.hpp>
#include <boost/logging/writer/ts_write.hpp>

using namespace boost::logging;

// Step 6: Define the filters and loggers you'll use
BOOST_DEFINE_LOG(g_l, log_type)
BOOST_DEFINE_LOG_FILTER(g_l_filter, level::holder)


void initLogs() {

    // Add formatters and destinations
    // That is, how the message is to be formatted...
    g_l()->writer().add_formatter( formatter::named_spacer("[%index%] %time% ")
        .add( "index", formatter::idx())
        .add( "time", formatter::time("$dd/$MM/$yyyy $hh:$mm.$ss")) );
    g_l()->writer().add_formatter( formatter::append_newline_if_needed() );

    //        ... and where should it be written to
    g_l()->writer().add_destination( destination::file("pipsqueak.log") );
    g_l()->writer().add_destination( destination::cerr() );
    g_l()->turn_cache_off();
}

