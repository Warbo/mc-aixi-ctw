#ifndef __AGENT_HPP__
#define __AGENT_HPP__

#include <iostream>

// boost includes
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/function.hpp>

#include "pipsqueak.hpp"

class FactoredContextTree;
class ContextTree;

// hash used to map entire history sequences to numbers
typedef uint64_t hash_t;

class ModelUndo;


class Agent {

    friend class boost::serialization::access;
    friend class ModelUndo; // memento

    public:

        /// construct a learning agent from the command line arguments
        Agent(size_t id);

        ~Agent(void);

        /// copy contructor / assignment operator that performs a deep copy
        explicit Agent(const Agent &rhs, size_t id);
        Agent &operator=(const Agent &rhs);

        /// construct a learning agent from a saved file
        Agent(const std::string &filename);

        /// save the state of an agent to disk
        bool save(const std::string &filename) const;

        /// serialization routine
        template<class Archive>
        void serialize(Archive &ar, const unsigned int version);

        /// select a legal action uniformly at random
        action_t selectRandomAction(randgen_t &rng) const;

        /// current age of the agent in cycles
        age_t age(void) const;

        /// number of distinct actions
        unsigned int numActions(void) const;

        /// hash of the entire history sequence
        hash_t hash(void) const;

        /// generate an action distributed according
        /// to our history statistics
        action_t genAction(randgen_t &rng) const;

        /// generate a percept distributed according
        /// to our history statistics
        void genPercept(randgen_t &rng, symbol_list_t &symlist) const;

        /// generate a percept distributed to our history statistics, and
        /// update our mixture environment model with it
        void genPerceptAndUpdate(randgen_t &rng, symbol_list_t &symlist);

        /// the total accumulated reward across an agents lifespan
        reward_t reward(void) const;

        /// the average reward received by the agent at each time step
        reward_t averageReward(void) const;

        /// maximum reward in a single time instant
        reward_t maxReward(void) const;

        /// minimum reward in a single time instant
        reward_t minReward(void) const;

        /// update the internal agent's model of the world
        /// due to receiving a percept or performing an action
        void modelUpdate(const symbol_list_t &percept);
        void modelUpdate(action_t action);

        /// revert the agent's internal model of the world
        /// to that of a previous time cycle, false on failure
        bool modelRevert(const ModelUndo &mu);

        /// the length of the stored history for an agent
        size_t historySize(void) const;

        /// determines the reward from a percept
        reward_t rewardFromPercept(const symbol_list_t &percept) const;

        /// log dynamic agent properties
        void logDynamicProperties(void) const;

        /// gather an action from the environment -> agent channel
        bool percept2symlist(const std::string &str, symbol_list_t &symlist) const;

        /// send an action across the agent -> environment channel
        void sendAction(action_t action, std::ostream &out) const;

        /// resets the agent
        void reset(void);

        /// length of the search horizon used by the agent
        size_t horizon(void) const;

        /// hash of history if we were to make a particular action
        hash_t hashAfterAction(action_t action) const;

        /// agent identification number
        size_t id(void) const;

        /// sets the function that computes the context
        void setContextFunctor(boost::function<void (context_t&)> functor);

        /// is the agent constructing an internal model of it's own behaviour
        bool useSelfModel(void) const;

        /// probability of selecting an action according to the
        /// agent's internal model of it's own behaviour
        double getPredictedActionProb(action_t action);

        /// get the agent's probability of receiving a particular percept
        double perceptProbability(const symbol_list_t &percept) const;

    private:

        // log the startup configuration properties of the agent
        void logStartUpProperties(void) const;

        // computes the resultant history hash after processing a set of symbols
        hash_t hashAfterSymbols(const symbol_list_t &new_syms) const;

        // computes the resultant history hash after processing a symbol
        hash_t hashAfterSymbol(symbol_t sym, hash_t hash) const;

        // convert a list of symbols to an action, false on failure
        bool symsToAction(const symbol_list_t &symlist, action_t &action) const;

        // converts a list of symbols to a reward, false on failure
        bool symsToReward(const symbol_list_t &symlist, reward_t &reward) const;

        // encodes an action into a block of symbols
        void encodeAction(action_t action, symbol_list_t &symlist) const;

        // action sanity check
        bool isActionOk(action_t action) const;

        // reward sanity check
        bool isRewardOk(reward_t reward) const;

        // deep copies the information within one agent to another
        void copyAgent(const Agent &rhs);

        // update the non-context tree part of an internal agent after receiving a percept
        void nonCTModelUpdate(const symbol_list_t &percept);

        // agent properties (update copy constructor if changed!)
        unsigned int m_actions_c;
        unsigned int m_actions_bits_c;
        unsigned int m_obs_bits_c;
        unsigned int m_rew_bits_c;
        size_t m_horizon;

        FactoredContextTree *m_ct;
        ContextTree *m_self_model;

        // incremental hash of the entire history sequence
        hash_t m_hash;

        uint64_t m_time_cycle;
        bool m_last_update_percept;
        bool m_base2_reward_encoding;
        reward_t m_total_reward;
        size_t m_id;

        bool m_use_self_model;
};


/* agent serialization routine - needs to be accessible from header */
template<class Archive>
inline void Agent::serialize(Archive &ar, const unsigned int version) {

    ar & m_actions_c;
    ar & m_actions_bits_c;
    ar & m_obs_bits_c;
    ar & m_rew_bits_c;
    ar & m_horizon;
    ar & m_ct;
    ar & m_self_model;
    ar & m_hash;
    ar & m_time_cycle;
    ar & m_last_update_percept;
    ar & m_base2_reward_encoding;
    ar & m_total_reward;
    ar & m_id;
    ar & m_use_self_model;
}


// used to store sufficient information to revert an agent
// to a copy of itself from a previous time cycle
class ModelUndo {

    public:
        /// construct a save point
        ModelUndo(const Agent &agent);

        /// saved state age accessor
        age_t age(void) const;

        /// saved state hash accessor
        hash_t hash(void) const;

        /// saved state reward accessor
        reward_t reward(void) const;

        /// saved state history size accessor
        size_t historySize(void) const;

        /// saved state accessor
        bool lastUpdatePercept(void) const;

    private:
        age_t m_age;
        hash_t m_hash;
        reward_t m_reward;
        size_t m_history_size;
        bool m_last_update_percept;
};


// a group of agents that share the same world model
class Hive {

    public:

        /// construct a hive of n agents
        Hive(size_t n);

        /// construct a hive of n indentical agents defined in a file
        Hive(size_t n, const std::string &filename);

        /// agent accessors
        Agent *operator[](size_t idx);
        const Agent *operator[](size_t idx) const;

        /// how many agents are in the hive
        size_t count(void) const;

        /// update the internal agents model of the world
        /// due to receiving a percept or performing an action
        void modelUpdate(const symbol_list_t &percept);
        void modelUpdate(action_t action);

    private:

        boost::ptr_vector<Agent> m_agents;
};


#endif // __AGENT_HPP__

