#ifndef __PIPSQUEAK_HPP__
#define __PIPSQUEAK_HPP__

#include <exception>
#include <vector>

// boost includes
#include <boost/cstdint.hpp>
#include <boost/random.hpp>

// fixed size integer types
using boost::int16_t;
using boost::uint16_t;
using boost::uint32_t;
using boost::int32_t;
using boost::uint64_t;
using boost::int64_t;


// program configuration options
namespace boost {
    namespace program_options {
        class variables_map;
    }
}
extern boost::program_options::variables_map options;


/// malformed environment response exception
class BadPerceptException : public std::exception {
    const char *what(void) const throw() { return "invalid percept"; }
};


/// out of time for search exception
class OutOfTimeException : public std::exception {
    const char *what(void) const throw() { return "out of time"; }
};


/// failure to allocate search node exception
class SearchNodeAllocFailException : public std::exception {
    const char *what(void) const throw() { return "could node allocate search node"; }
};


// symbols that can be predicted
enum symbol_t {
    Off = 0,
    On  = 1
};

// a list of symbols
typedef std::vector<symbol_t> symbol_list_t;

// a representation of a context
typedef std::vector<symbol_t> context_t;

// describe the reward accumulated by an agent
typedef double reward_t;

// describe the age of an agent
typedef uint64_t age_t;

// describes an agent action
typedef unsigned int action_t;

// random number generator to supply noise
typedef boost::mt19937 randsrc_t;
typedef boost::uniform_01<randsrc_t> randgen_t;


#endif // __PIPSQUEAK_HPP__

