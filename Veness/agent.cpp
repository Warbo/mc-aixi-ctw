#include "agent.hpp"

#include <iostream>
#include <string>
#include <sstream>
#include <exception>
#include <fstream>
#include <cassert>

// boost includes
#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/program_options.hpp>

#include "pipsqueak.hpp"
#include "log.hpp"
#include "search.hpp"
#include "predict.hpp"


/* simple destructor */
Agent::~Agent(void) {

    if (m_ct) delete m_ct;
    if (m_self_model) delete m_self_model;
}


/* construct a learning agent that interacts across a two way binary channel */
Agent::Agent(size_t id) {

    // setup agent configurating from command line arguments
    m_horizon = options["agent-horizon"].as<unsigned int>();

    m_rew_bits_c  = options["reward-bits"].as<unsigned int>();
    m_obs_bits_c  = options["observation-bits"].as<unsigned int>();
    m_actions_c   = options["agent-actions"].as<unsigned int>();

    m_ct = new FactoredContextTree(
      m_obs_bits_c + m_rew_bits_c,
      options["ct-depth"].as<unsigned int>()
    );
    m_self_model = NULL;

    m_base2_reward_encoding = options["reward-encoding"].as<std::string>() == "base2";

    m_use_self_model = options.count("bootstrapped-playouts") > 0;
    if (m_use_self_model) {
        m_self_model = new ContextTree(options["ct-depth"].as<unsigned int>());
    }

    // find the number of bits needed to output the action
    for (unsigned int i = 1, c = 1; i < m_actions_c; i *= 2, c++) {
        m_actions_bits_c = c;
    }

    m_id = id;

    reset();
}


/* deep copies the information within one agent to another */
void Agent::copyAgent(const Agent &rhs) {

    m_actions_bits_c        = rhs.m_actions_bits_c;
    m_actions_c             = rhs.m_actions_c;
    m_base2_reward_encoding = rhs.m_base2_reward_encoding;
    m_ct                    = new FactoredContextTree(*rhs.m_ct);
    m_self_model            = (rhs.m_self_model != NULL) ?
                                new ContextTree(*rhs.m_self_model) : NULL;
    m_hash                  = rhs.m_hash;
    m_horizon               = rhs.m_horizon;
    m_last_update_percept   = rhs.m_last_update_percept;
    m_obs_bits_c            = rhs.m_obs_bits_c;
    m_rew_bits_c            = rhs.m_rew_bits_c;
    m_time_cycle            = rhs.m_time_cycle;
    m_total_reward          = rhs.m_total_reward;
    m_id                    = rhs.m_id;
    m_use_self_model        = rhs.m_use_self_model;
}


/* agent copy contructor that performs a deep copy */
Agent::Agent(const Agent &rhs, size_t id) {

    copyAgent(rhs);
    m_id = id;
}


/* agent assignment operator that performs a deep copy */
Agent &Agent::operator=(const Agent &rhs) {

    if (this != boost::addressof(rhs)) {
        if (m_ct != NULL) delete m_ct;
        copyAgent(rhs);
    }

    return *this;
}


/* construct a learning agent from a saved file */
Agent::Agent(const std::string &filename) {

    if (options.count("binary-io") > 0) {

        // load the agent from file
        std::ifstream ifs(filename.c_str(), std::ios_base::binary);
        if (!ifs) throw std::invalid_argument("bad agent file.");
        LINFO_ << "loading (binary-mode) agent from: " << filename << std::endl;

        boost::archive::binary_iarchive ia(ifs);
        ia >> *this;

    } else {

        // load the agent from file
        std::ifstream ifs(filename.c_str());
        if (!ifs) throw std::invalid_argument("bad agent file.");
        LINFO_ << "loading (text-mode) agent from: " << filename << std::endl;

        boost::archive::text_iarchive ia(ifs);
        ia >> *this;
    }

    logStartUpProperties();
}


/* resets the agent */
void Agent::reset(void) {

    m_ct->clear();
    if (m_use_self_model) m_self_model->clear();

    // initialise DJB2-SDBM hash
    m_hash = hash_t(5381) << 32;

    m_time_cycle = 0;
    m_last_update_percept = false;
    m_total_reward = 0.0;

    logStartUpProperties();
}


/* log the startup configuration properties of the agent. */
void Agent::logStartUpProperties(void) const {

    std::string header = id() ? "Additional agent with id=" : "Agent ";

    LINFO_ << header << id() << " created.";
    if (id()) return;

    LINFO_ << "Channel properties: (O=" << m_obs_bits_c << ", R=" << m_rew_bits_c
        << " [" << (m_base2_reward_encoding ? "base2" : "bitcount") << "] "
           << "A=" << m_actions_bits_c << ")" << std::endl;
    LINFO_ "Age=" << age() << ", Search Horizon=" << horizon() << std::endl;
    LINFO_ << "Context Tree nodes: " << m_ct->size();
}


/* log the dynamic agent properties */
void Agent::logDynamicProperties(void) const {

    LINFO_ << "Average reward per time cycle: " << averageReward();
    LINFO_ << "Accumulated reward: " << reward() << std::endl;
    LINFO_ << "History Hash: " << hash() << " Agent age: " << age() << std::endl;
}


/* converts a percept string into a list of symbols, false on failure */
bool Agent::percept2symlist(const std::string &str, symbol_list_t &symlist) const {

    symlist.clear();

    unsigned int percept_len = m_obs_bits_c + m_rew_bits_c;

    if (str.length() != percept_len) return false;

    for (unsigned int i = 0; i < percept_len; i++) {
        if (str[i] != '1' && str[i] != '0') return false;
        symbol_t sym = (str[i] - '0') ? On : Off;
        symlist.push_back(sym);
    }

    return true;
}


/* select an action uniformly at random */
action_t Agent::selectRandomAction(randgen_t &rng) const {

    unsigned int rval = static_cast<unsigned int>(rng()*m_actions_c);
    assert(rval >=0 && rval < m_actions_c);

    return rval;
}


/* select an action distributed according to our history statistics */
action_t Agent::genAction(randgen_t &rng) const {

    symbol_list_t syms;
    action_t action;

    // use rejection sampling to pick an action according
    // to our historical distribution
    do {
        m_self_model->genRandomSymbols(rng, syms, m_actions_bits_c);
    } while (!symsToAction(syms, action));

    return action;
}


/* generate a percept distributed according to our history statistics */
void Agent::genPercept(randgen_t &rng, symbol_list_t &symlist) const {

    m_ct->genRandomSymbols(rng, symlist, m_obs_bits_c + m_rew_bits_c);
}


/* generate a percept distributed to our history statistics, and
   update our internal agent state. this is more efficient than calling
   genPercept and modelUpdate separately. */
void Agent::genPerceptAndUpdate(randgen_t &rng, symbol_list_t &symlist) {

    m_ct->genRandomSymbolsAndUpdate(rng, symlist, m_obs_bits_c + m_rew_bits_c);
    nonCTModelUpdate(symlist);
}


/* return the total accumulated reward across an agents lifespan */
reward_t Agent::reward(void) const {

    return m_total_reward;
}


/* encodes an action into a block of symbols */
void Agent::encodeAction(action_t action, symbol_list_t &symlist) const {

    assert(isActionOk(action));

    symlist.clear();

    for (unsigned int i = 0; i < m_actions_bits_c; i++) {
        symbol_t sym = (action & (1 << (m_actions_bits_c - i - 1))) ? On : Off;
        symlist.push_back(sym);
    }
}


/* send an action across the agent -> environment channel */
void Agent::sendAction(action_t action, std::ostream &out) const {

    assert(isActionOk(action));

    std::ostringstream oss;
    symbol_list_t symlist;

    encodeAction(action, symlist);

    action_t inv_action;
    symsToAction(symlist, inv_action);
    assert(inv_action == action);

    symbol_list_t::iterator it = symlist.begin();
    for (; it != symlist.end(); ++it) {
        oss << ((*it    == On) ? "1" : "0");
    }

    LINFO_ << "encoded action: " << oss.str() << std::endl;

    out << oss.str() << std::endl;
}


/* computes the resultant history hash after processing a single symbol */
hash_t Agent::hashAfterSymbol(symbol_t sym, hash_t hash) const {

   hash_t c = (sym == On) ? '1' : '0';

   // update with a single iteration of the SDBM hash
   hash_t low = (hash << 32) >> 32;
   low = c + (low << 6) + (low << 16) - low;

   // update with a single iteration of the DJB2 hash
   hash_t high = hash >> 32;
   high = ((high << 5) + high) + c;

   // combine
   return (high << 32) | low;
}


/* computes the resultant history hash after processing a set of symbols */
hash_t Agent::hashAfterSymbols(const symbol_list_t &new_syms) const {

    hash_t rval = m_hash;

    // update the hash of the history
    symbol_list_t::const_iterator it = new_syms.begin();
    for (; it != new_syms.end(); ++it) {
        rval = hashAfterSymbol(*it, rval);
    }

    return rval;
}


/* update the internal agent's model of the world due to receiving a percept */
void Agent::modelUpdate(const symbol_list_t &percept) {

    assert(percept.size() == m_obs_bits_c + m_rew_bits_c);

    m_ct->update(percept);
    nonCTModelUpdate(percept);
}


/* update the non-context tree part of an internal agent after receiving a percept */
void Agent::nonCTModelUpdate(const symbol_list_t &percept) {

    if (m_use_self_model) m_self_model->updateHistory(percept);

    m_hash = hashAfterSymbols(percept);
    m_total_reward += rewardFromPercept(percept);
    m_last_update_percept = true;
}


/* update the internal agent's model of the world due to receiving an action */
void Agent::modelUpdate(action_t action) {

    assert(isActionOk(action));
    assert(m_last_update_percept == true);

    symbol_list_t action_syms;
    encodeAction(action, action_syms);

    m_ct->updateHistory(action_syms);

    // update the agent's internal heuristic of it's own behaviour
    if (m_use_self_model) m_self_model->update(action_syms);

    m_hash = hashAfterSymbols(action_syms);

    m_time_cycle++;
    m_last_update_percept = false;
}


/* hash of history if we were to make a particular action */
hash_t Agent::hashAfterAction(action_t action) const {

    assert(isActionOk(action));

    symbol_list_t action_syms;
    encodeAction(action, action_syms);

    return hashAfterSymbols(action_syms);
}


/* current age of the agent in cycles */
age_t Agent::age(void) const {

    return m_time_cycle;
}


/* hash of the entire history sequence */
hash_t Agent::hash(void) const {

    return m_hash;
}


/* save the state of an agent to disk */
bool Agent::save(const std::string &filename) const {

    if (options.count("binary-io") > 0) {

        std::ofstream ofs(filename.c_str(), std::ios_base::binary);
        if (!ofs) return false;

        boost::archive::binary_oarchive oa(ofs);
        oa << *this;

    } else {

        std::ofstream ofs(filename.c_str());
        if (!ofs) return false;

        boost::archive::text_oarchive oa(ofs);
        oa << *this;
    }

    LINFO_ << "successfully saved agent to: " << filename;

    return true;
}


/* convert a list of symbols to an action, false on failure */
bool Agent::symsToAction(const symbol_list_t &symlist, action_t &action) const {

    action = 0;

    symbol_list_t::const_reverse_iterator it = symlist.rbegin();
    for (action_t c = 0; it != symlist.rend(); ++it, c++) {
        if (*it == On) action |= (1 << c);
    }

    return isActionOk(action);
}


/* converts a list of symbols to a reward, false on failure */
bool Agent::symsToReward(const symbol_list_t &symlist, reward_t &reward) const {

    assert(symlist.size() == m_rew_bits_c);

    symbol_list_t::const_reverse_iterator it = symlist.rbegin();

    if (m_base2_reward_encoding) { // base2 reward encoding

        unsigned int r = 0;
        for (unsigned int c = 0; c < m_rew_bits_c; ++it, c++) {
            if (*it == On) r |= (1 << c);
        }
        reward = reward_t(r);

    } else {  // bitcount encoding

        reward = 0.0;
        for (unsigned int c = 0; c < m_rew_bits_c; ++it, c++) {
            if (*it == On) reward += 1.0;
        }
    }

    return isRewardOk(reward);
}


/* interprets a list of symbols as a reward */
reward_t Agent::rewardFromPercept(const symbol_list_t &percept) const {

    assert(percept.size() == m_obs_bits_c + m_rew_bits_c);

    symbol_list_t::const_reverse_iterator it = percept.rbegin();

    if (m_base2_reward_encoding) { // base2 reward encoding

        unsigned int r = 0;

        for (unsigned int c = 0; c < m_rew_bits_c; ++it) {
            assert(it != percept.rend());
            if (*it == On) r |= (1 << c);
            c++;
        }
        return reward_t(r);
    }

    // assume the reward is the number of on bits
    double reward = 0.0;

    for (unsigned int c = 0; c < m_rew_bits_c; ++it) {
        assert(it != percept.rend());
        if (*it == On) reward += 1.0;
        c++;
    }

    return reward;
}


/* maximum or minimum reward in a single time cycle */
reward_t Agent::maxReward(void) const {

    double mr = double(m_rew_bits_c);
    if (m_base2_reward_encoding) mr = std::pow(2.0, mr) - 1.0;

    return mr;
}


/* minimum reward in a single time cycle */
reward_t Agent::minReward(void) const {

    return 0.0;  // TODO: should we allow negative reward?
}


/* number of distinct actions */
unsigned int Agent::numActions(void) const {

    return m_actions_c;
}


/* agent identification number */
size_t Agent::id(void) const {

    return m_id;
}


/* revert the agent's internal model of the world
   to that of a previous time cycle, false on failure */
bool Agent::modelRevert(const ModelUndo &mu) {

    assert(m_ct->historySize() > mu.historySize());
    assert(!m_use_self_model || m_self_model->historySize() > mu.historySize());

    if (m_time_cycle < mu.age()) return false;

    // agent properties must be reverted before context update,
    // since the predicates that depend on the context may
    // depend on them
    m_time_cycle          = mu.age();
    m_hash                = mu.hash();
    m_total_reward        = mu.reward();
    m_last_update_percept = mu.lastUpdatePercept();

    // revert the context tree and history back to it's previous state

    if (mu.lastUpdatePercept()) { // if we are undoing an action
        m_ct->revertHistory(mu.historySize());
        if (m_use_self_model) {
            size_t end_size = m_self_model->historySize();
            for (size_t i = 0; i < end_size - mu.historySize(); i++) {
                m_self_model->revert();
            }
        }
    } else { // if we are undoing an observation / reward
        size_t end_size = m_ct->historySize();
        size_t percept_bits = m_obs_bits_c + m_rew_bits_c;
        for (size_t i = 0; i < end_size - mu.historySize(); i++) {
            m_ct->revert(percept_bits - i - 1);
        }
        if (m_use_self_model) m_self_model->revertHistory(mu.historySize());
    }

    assert(!m_use_self_model || m_self_model->historySize() == m_ct->historySize());

    return true;
}


/* the length of the stored history for an agent */
size_t Agent::historySize(void) const {

    return m_ct->historySize();
}


/* length of the search horizon used by the agent */
size_t Agent::horizon(void) const {

    return m_horizon;
}


/* action sanity check */
bool Agent::isActionOk(action_t action) const {

    return action < m_actions_c;
}


/* reward sanity check */
bool Agent::isRewardOk(reward_t reward) const {

    return reward >= minReward() && reward <= maxReward();
}


/* sets the function that computes the context */
void Agent::setContextFunctor(boost::function<void (context_t&)> functor) {

    m_ct->setContextFunctor(functor);
}


/* the average reward received by the agent at each time step */
reward_t Agent::averageReward(void) const {

    return age() > 0 ? reward() / reward_t(age()) : 0.0;
}


/* is the agent constructing an internal model of it's own behaviour */
bool Agent::useSelfModel(void) const {

    return m_use_self_model;
}


/* probability of selecting an action according to the
   agent's internal model of it's own behaviour */
double Agent::getPredictedActionProb(action_t action) {

    // actions are equally likely if no internal model is used
    if (!m_use_self_model) return 1.0 / double(m_actions_c);

    // compute normalisation term, since some
    // actions may be illegal
    double tot = 0.0;
    symbol_list_t symlist;

    for (action_t a=0; a < m_actions_c; a++) {
        encodeAction(a, symlist);
        tot += m_self_model->predict(symlist);
    }

    assert(tot != 0.0);
    encodeAction(action, symlist);
    return m_self_model->predict(symlist) / tot;
}


/* get the agent's probability of receiving a particular percept */
double Agent::perceptProbability(const symbol_list_t &percept) const {

    assert(percept.size() == m_obs_bits_c + m_rew_bits_c);
    return m_ct->predict(percept);
}


/* used to revert an agent to a previous state */
ModelUndo::ModelUndo(const Agent &agent) {

    m_age          = agent.age();
    m_hash         = agent.hash();
    m_reward       = agent.reward();
    m_history_size = agent.historySize();

    m_last_update_percept = agent.m_last_update_percept;
}


/* previous agent time cycle */
age_t ModelUndo::age(void) const {

    return m_age;
}


/* previous time cycle history hash */
hash_t ModelUndo::hash(void) const {

    return m_hash;
}


/* previous time cycle accumulated reward */
reward_t ModelUndo::reward(void) const {

    return m_reward;
}


/* previous time cycle history length */
size_t ModelUndo::historySize(void) const {

    return m_history_size;
}


/* was the last update a percept? */
bool ModelUndo::lastUpdatePercept(void) const {

    return m_last_update_percept;
}


/* construct a hive of n agents */
Hive::Hive(size_t n) {

    for (size_t i=0; i < n; i++) {
        m_agents.push_back(new Agent(i));
    }
}


/* construct a hive of n indentical agents defined in a file */
Hive::Hive(size_t n, const std::string &filename) {

    m_agents.push_back(new Agent(filename));

    for (size_t i=1; i < n; i++) {
        m_agents.push_back(new Agent(m_agents[0], i));
    }
}


/* non-const agent accessor, NULL if no such agent exists */
Agent *Hive::operator[](size_t idx) {

    return idx < m_agents.size() ? &m_agents[idx] : NULL;
}


/* const agent accessor, NULL if no such agent exists */
const Agent *Hive::operator[](size_t idx) const {

    return idx < m_agents.size() ? &m_agents[idx] : NULL;
}


/* how many agents are in the hive */
size_t Hive::count(void) const {

    return m_agents.size();
}


/* update the internal agents model of the world
   due to receiving a percept or performing an action */
void Hive::modelUpdate(const symbol_list_t &percept) {

    for (size_t i=0; i < m_agents.size(); i++) {
        m_agents[i].modelUpdate(percept);
    }
}


/* update the internal agents model of the world
   due to receiving a percept or performing an action */
void Hive::modelUpdate(action_t action) {

    for (size_t i=0; i < m_agents.size(); i++) {
        m_agents[i].modelUpdate(action);
    }
}

