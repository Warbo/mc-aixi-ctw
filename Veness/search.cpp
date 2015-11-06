#include "search.hpp"

#include <exception>
#include <limits>
#include <cstdlib>
#include <cmath>

// boost includes
#include <boost/random.hpp>
#include <boost/program_options.hpp>
#include <boost/timer.hpp>
#include <boost/thread/thread.hpp>

#include "pipsqueak.hpp"
#include "agent.hpp"
#include "log.hpp"

#undef max

typedef uint64_t visits_t;


// search options
static const visits_t     MinVisitsBeforeExpansion = 1;
static const unsigned int MaxDistanceFromRoot  = 100;
static size_t             MaxSearchNodes;

// shared (not-threadsafe) uniform 0/1 prng
static randsrc_t rng;
static randgen_t urng(rng);


// contains information about a single "state"
class SearchNode {

    public:

        SearchNode(hash_t hash, bool is_chance_node);

        /// determine the next action to play
        action_t selectAction(Agent &agent, randgen_t &gen) const;

        /// determine the expectated reward from this node
        reward_t expectation(void) const;

        /// perform a sample run through this node and it's children,
        /// returning the accumulated reward from this sample run
        reward_t sample(Agent &agent, randgen_t &rng, unsigned int dfr);

        /// the hash code of the history represented by this search node
        hash_t hash(void) const;

        /// number of times the seach node has been visited
        visits_t visits(void) const;

    private:

        bool m_chance_node;
        double m_mean;
        visits_t m_visits_c;
        hash_t m_hash;

        mutable boost::mutex m_mutex;
};


// local prototypes
static reward_t playout(Agent &agent, randgen_t &rng, unsigned int playout_len);
static SearchNode *findOrCreateNode(hash_t hash, bool is_chance_node);
static SearchNode *findNode(hash_t hash);
static void destroyNode(SearchNode *n);
static size_t garbageCollect(SearchNode *tree);


// memory pool used by the search
typedef std::map<hash_t, SearchNode *> search_node_pool_t;
static search_node_pool_t search_node_pool;

static SearchNode *gbl_search_root = NULL;
static boost::mutex m_mempool_mutex; // global lock for memory routines


/* create a new search node */
SearchNode::SearchNode(hash_t hash, bool is_chance_node) {
    m_chance_node = is_chance_node;
    m_mean        = 0.0;
    m_visits_c    = 0;
    m_hash        = hash;
}


/* the hash code of the history represented by this search node */
hash_t SearchNode::hash(void) const {

    return m_hash;
}


/* determine the expected reward from this node until the horizon */
reward_t SearchNode::expectation(void) const {

    return m_mean;
}


/* number of times the seach node has been visited */
visits_t SearchNode::visits(void) const {

    return m_visits_c;
}


/* perform a sample run through this node and it's children,
   returning the accumulated reward from this sample run */
reward_t SearchNode::sample(Agent &agent, randgen_t &rng, unsigned int dfr) {

    if (dfr == agent.horizon() * 2) return 0.0;

    ModelUndo undo(agent);
    reward_t reward = 0.0;

    if (m_chance_node) {  // handle chance nodes

        // generate a hypothetical percept
        symbol_list_t percept;
        agent.genPerceptAndUpdate(rng, percept);

        // extract the reward for this transition, and
        // update the agent model
        reward = agent.rewardFromPercept(percept);

        SearchNode *n = findOrCreateNode(agent.hash(), false);
        reward += n->sample(agent, rng, dfr + 1);
        agent.modelRevert(undo);

    } else {  // handle decision nodes

        m_mutex.lock();

        // if we need to do a playout
        bool do_playout =
            visits() < MinVisitsBeforeExpansion           ||
            dfr >= MaxDistanceFromRoot                    ||
            search_node_pool.size() >= MaxSearchNodes;

        if (do_playout) {

            m_mutex.unlock();

            reward = playout(agent, rng, agent.horizon() - dfr / 2);

        } else {

            // pick an action
            action_t a = selectAction(agent, rng);
            m_mutex.unlock();

            // update model, and recurse
            agent.modelUpdate(a);
            SearchNode *n = findOrCreateNode(agent.hash(), true);
            reward = n->sample(agent, rng, dfr + 1);
            agent.modelRevert(undo);
        }
    }

    { // update our statistics for this node
        boost::mutex::scoped_lock lock(m_mutex);
        double vc = double(m_visits_c);
        m_mean = (m_mean * vc + reward) / (vc + 1.0);
        m_visits_c++;
    }

    return reward;
}


/* determine the next child to explore, NULL if no such child exists */
action_t SearchNode::selectAction(Agent &agent, randgen_t &rng) const {

    // higher values encourage more exploration, less exploitation
    const double ExploreBias = agent.horizon() * agent.maxReward();
    const double UnexploredBias = 1000000000.0;

    assert(!m_chance_node);

    action_t best_action;
    double best_priority = -std::numeric_limits<double>::infinity();

    for (action_t a=0; a < agent.numActions(); a++) {

        SearchNode *n = findNode(agent.hashAfterAction(a));
        assert(n == NULL || n->m_chance_node);

        double priority, noise = rng() * 0.0001;

        // use UCB formula to compute priority
        if (n == NULL || n->visits() == 0) {
            priority =  UnexploredBias + noise;
        } else {
            double pvisits = double(visits());
            double cvisits = double(n->visits());
            double bias = ExploreBias * std::sqrt(2.0 * std::log(pvisits) / cvisits);
            priority = n->expectation() + bias + noise;
        }

        if (priority > best_priority) {
            best_action = a;
            best_priority = priority;
        }
    }

    return best_action;
}


/* attempts to find a node in the search tree */
static SearchNode *findNode(hash_t hash) {

    boost::mutex::scoped_lock lock(m_mempool_mutex);

    search_node_pool_t::iterator it = search_node_pool.find(hash);
    return it != search_node_pool.end() ? it->second : NULL;
}


/* attempts to find a node in the search tree, otherwise a new node is made */
static SearchNode *findOrCreateNode(hash_t hash, bool is_chance_node) {

    boost::mutex::scoped_lock lock(m_mempool_mutex);

    search_node_pool_t::iterator it = search_node_pool.find(hash);
    if (it != search_node_pool.end()) return it->second;

    SearchNode *rval = new SearchNode(hash, is_chance_node);
    search_node_pool[hash] = rval;

    return rval;
}


/* remove a node from the search tree */
static void destroyNode(SearchNode *n) {

    boost::mutex::scoped_lock lock(m_mempool_mutex);

    assert(search_node_pool.find(n->hash()) != search_node_pool.end());

    search_node_pool.erase(n->hash());
    if (n == gbl_search_root) gbl_search_root = NULL;
}


/* reclaim some memory by recursively freeing up space from old nodes,
   returning the number of nodes freed. */
static size_t garbageCollect(SearchNode *tree) {

    boost::mutex::scoped_lock lock(m_mempool_mutex);

    // TODO
    return 0;
}


/* simulate a path through a hypothetical future for the agent
   within it's internal model of the world, returning the
   accumulated reward. */
static reward_t playout(Agent &agent, randgen_t &rng, unsigned int playout_len) {

    reward_t start_reward = agent.reward();

    boost::ptr_vector<ModelUndo> undos;

    for (unsigned int i=0; i < playout_len; i++) {

        undos.push_back(new ModelUndo(agent));

        // generate action
        action_t a = agent.useSelfModel() ?
            agent.genAction(rng) : agent.selectRandomAction(rng);
        agent.modelUpdate(a);

        // generate percept
        symbol_list_t percept;
        undos.push_back(new ModelUndo(agent));
        agent.genPerceptAndUpdate(rng, percept);
    }

    reward_t rval = agent.reward() - start_reward;

    boost::ptr_vector<ModelUndo>::reverse_iterator it = undos.rbegin();
    for (; it != undos.rend(); ++it) {
        agent.modelRevert(*it);
    }

    return rval;
}


/* determine the best action with naive 1-ply monte-carlo sampling */
static action_t naiveMonteCarlo(Agent &agent)  {

    boost::timer ti;

    // determine the depth and number of seconds to search
    double time_limit_ms = double(options["cycle-length-ms"].as<unsigned int>());
    double time_limit = time_limit_ms / 1000.0;

    // sufficient statistics to compute the sample mean for each action
    std::vector<std::pair<reward_t, double> > r(agent.numActions());

    for (unsigned int i = 0; i < agent.numActions(); i++) {
        r[i].first = r[i].second = 0.0;
    }

    ModelUndo mu(agent);
    size_t total_samples = 0;
    size_t start_hist_len = agent.historySize();

    do {  // we ensure each action always has one estimate
        for (action_t i = 0; i < agent.numActions(); i++) {

            // make action
            agent.modelUpdate(i);

            // grab percept and determine immediate reward
            symbol_list_t percept;
            agent.genPerceptAndUpdate(urng, percept);
            reward_t reward = agent.rewardFromPercept(percept);

            // playout the remainder of the sequence
            reward += playout(agent, urng, agent.horizon() - 1);

            r[i].first  += reward;
            r[i].second += 1.0;

            agent.modelRevert(mu);
            assert(start_hist_len == agent.historySize());

            total_samples++;
        }
    } while (ti.elapsed() < time_limit);

    // determine best arm, breaking ties arbitrarily
    double best = -std::numeric_limits<double>::infinity();
    action_t best_action = 0;
    for (unsigned int i = 0; i < agent.numActions(); i++) {

        assert(r[i].second > 0.0);
        double noise = urng() * 0.0001;

        double x = r[i].first / r[i].second + noise;

        if (x > best) {
            best = x;
            best_action = i;
        }
    }

    LINFO_ << "naive monte-carlo decision based on " << total_samples << " samples.";

    for (unsigned int i=0; i < agent.numActions(); i++) {
        LDBG_ << "action " << i << ": " << r[i].first / r[i].second;
    }

    return best_action;
}


/* initialise the monte-carlo tree search */
static void initMCTS(void) {

    size_t mem_avail_bytes = options["memsearch"].as<size_t>() * 1024 * 1024;

    // we assume overhead for each node is upper bounded by:
    size_t slot_size = sizeof(search_node_pool_t::key_type)   +
                       sizeof(search_node_pool_t::value_type) +
                       sizeof(void *) * 4;

    MaxSearchNodes = mem_avail_bytes / slot_size;

    // determine how much memory to use for the search
    LINFO_ << "using " << options["memsearch"].as<size_t>()
           << "MB for search pool of " << MaxSearchNodes << " nodes. " << std::endl;

    if (options.count("bootstrapped-playouts") > 0) {
        LINFO_ << "using bootstrapped playouts";
    }

    // TODO: keep relevant information from previous time cycle
    search_node_pool_t::iterator it = search_node_pool.begin();
    for (; it != search_node_pool.end(); ++it) {
        delete it->second;
    }
    search_node_pool.clear();
}


// wrapper class for each sampling thread
class TreeSampler {

    public:

        TreeSampler(Agent &agent, boost::timer &timer, double time_limit_s, size_t sims);

        /// prepare the multi-threaded search
        static void init(void);

        /// number of completed samples
        static uint64_t samples(void);

        /// grow the mcts tree by successive sampling
        void operator()(void);

    private:

        Agent &m_agent;
        randsrc_t m_rng;
        randgen_t m_gen;
        boost::timer &m_timer;
        double m_time_limit_s;
        size_t m_num_simulations;

        static boost::mutex m_stat_mutex;
        static uint64_t m_samples_c;
};

boost::mutex TreeSampler::m_stat_mutex;
uint64_t TreeSampler::m_samples_c = 0;


/* construct a single mcts worker */
TreeSampler::TreeSampler(Agent &agent, boost::timer &timer, double time_limit_s, size_t sims) :
    m_agent(agent),
    m_rng(),
    m_gen(m_rng),
    m_timer(timer),
    m_time_limit_s(time_limit_s),
    m_num_simulations(sims)
{
}


/* prepare the multi-threaded search */
void TreeSampler::init(void) {
    boost::mutex::scoped_lock lock(m_stat_mutex);
    m_samples_c = 0;
}


/* number of completed samples */
uint64_t TreeSampler::samples(void) {

    return m_samples_c;
}


/* grow the mcts game tree by successive sampling */
void TreeSampler::operator()(void) {

    ModelUndo mu(m_agent);

    try {
        do {
            {
                boost::mutex::scoped_lock lock(m_stat_mutex);
                if (m_samples_c >= m_num_simulations) break;
                m_samples_c++;
            }
            gbl_search_root->sample(m_agent, m_gen, 0);

        } while (m_timer.elapsed() < m_time_limit_s);

    } catch (OutOfTimeException &) {

        // TODO
    }
}


/* selects the best action determined by the MCTS statistics */
static action_t selectBestMCTSAction(Agent &agent) {

    ModelUndo mu(agent);
    action_t best_action = agent.selectRandomAction(urng);
    reward_t best_exp = -std::numeric_limits<reward_t>::infinity();
    bool found = false;

    for (action_t a=0; a < agent.numActions(); a++) {

        SearchNode *n = findNode(agent.hashAfterAction(a));

        if (n != NULL) {

            double noise = urng() * 0.0001;

            reward_t exp = n->expectation() + noise;
            if (exp > best_exp) {
                best_exp    = n->expectation();
                best_action = a;
                found = true;
            }

            LDBG_ << "action " << a << ": " << exp << "  visits: " << n->visits()
                  << "  self-predicted probability: " << agent.getPredictedActionProb(a);
        }
    }

    assert(found);
    return best_action;
}


/* perform a monte-carlo tree search to determine the best action */
static action_t mcts(Hive &hive) {

    Agent &agent = *hive[0];

    initMCTS();

    // determine number of seconds to search for
    double time_limit = std::numeric_limits<double>::max();
    if (options.count("cycle-length-ms")) {
        double time_limit_ms = double(options["cycle-length-ms"].as<unsigned int>());
        time_limit = time_limit_ms / 1000.0;
    }

    size_t sims = std::numeric_limits<size_t>::max();
    if (options.count("mc-simulations")) {
        sims = options["mc-simulations"].as<size_t>();
    }

    boost::timer timer; // and start counting

    // grab the root node of the search tree
    gbl_search_root = findOrCreateNode(agent.hash(), false);
    if (gbl_search_root == NULL) throw SearchNodeAllocFailException();

    // prepare to invoke the threads
    TreeSampler::init();
    size_t max_threads = options["threads"].as<unsigned int>();
    std::vector<TreeSampler *> threadpool(max_threads);

    // allocate the worker functors
    boost::thread_group threads;
    for (size_t i = 0; i < max_threads; i++) {
        threadpool[i] = new TreeSampler(*hive[i], timer, time_limit, sims);
    }

    // fire off the threads
    for (size_t i = 0; i < max_threads; i++) {
        threads.create_thread(boost::ref(*threadpool[i]));
    }

    // wait until they have all finished working
    threads.join_all();

    // reclaim the memory used by the threads
    for (size_t i = 0; i < max_threads; i++) delete threadpool[i];

    LINFO_ << "mcts decision based on " << TreeSampler::samples() << " samples.";

    return selectBestMCTSAction(agent);
}


/* determine the best action by simulating many
   possible future lives of the agent */
action_t search(Hive &hive) {

    boost::timer ti;

    Agent &agent = *hive[0];

    // select control algorithm
    action_t best;

    if (options["controller"].as<std::string>() == "mc") {
        best = naiveMonteCarlo(agent);
    } else if (options["controller"].as<std::string>() == "mcts") {
        best = mcts(hive);
    } else {
        best = agent.selectRandomAction(urng);
    }

    LINFO_ << "searched " << ti.elapsed() << "s for a best action." << std::endl;

    return best;
}

