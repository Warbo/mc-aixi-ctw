#ifndef __PROTOCOL_HPP__
#define __PROTOCOL_HPP__

#include <iostream>

class Agent;

/// agent <-> environment main-loop
extern void mainLoop(std::istream &in, std::ostream &out);

/// gets the agent by identification number, NULL on failure
extern Agent *getAgent(size_t id);

#endif // __PROTOCOL_HPP__


