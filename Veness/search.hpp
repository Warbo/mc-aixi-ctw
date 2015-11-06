#ifndef __SEARCH_HPP__
#define __SEARCH_HPP__

#include "pipsqueak.hpp"

class Hive;

/// determine the best action by searching ahead
extern action_t search(Hive &hive);

#endif // __SEARCH_HPP__

