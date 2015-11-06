#include "predict.hpp"

#include <vector>
#include <cassert>
#include <stack>
#include <iostream>
#include <cmath>

// boost includes
#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/utility.hpp>
#include <boost/thread/thread.hpp>

#include "pipsqueak.hpp"


// log KT multiplier cache options
static const bool UseLogKTMulCache    = true;
static const size_t LogKTMulCacheSize = 256;

static const symbol_t symbols[] = { Off, On };

// precompute std::log(0.5)
static const double log_point_five = std::log(0.5);


// KTMulCache - indexed by symbol counts "a and a+b"
static double log_kt_mul_cache[LogKTMulCacheSize][LogKTMulCacheSize];
static bool log_kt_mul_cache_initialized = false;
static boost::mutex log_kt_mul_cache_mutex;


/* precompute the log-kt-multiplier cache entries */
static void precomputeLogKTMulCache(void) {

    boost::mutex::scoped_lock f_lock(log_kt_mul_cache_mutex);

    if (!UseLogKTMulCache || log_kt_mul_cache_initialized) return;

    for (size_t i=0; i < LogKTMulCacheSize; i++) {
        for (size_t j=0; j < LogKTMulCacheSize; j++) {
            double kt_mul_numer = double(i) + 0.5;
            double kt_mul_denom = double(j + 1);
            log_kt_mul_cache[i][j] = std::log(kt_mul_numer / kt_mul_denom);
        }
    }

    log_kt_mul_cache_initialized = true;
}


/* flips a symbol type */
static symbol_t flip(symbol_t sym) {
    return sym == On ? Off : On;
}


/* compute the current context */
void ContextTree::getContext(context_t &context) const {

    if (!m_context_functor.empty()) {
        m_context_functor(context);
        return;
    }

    context.clear();

    history_t::const_reverse_iterator ri = m_history.rbegin();
    for (size_t c = 0; ri != m_history.rend() && c < m_depth; ++ri, c++) {
        context.push_back(*ri);
    }
}


/* create a new context node */
CTNode::CTNode(void) :
    m_log_prob_est(0.0),
    m_log_prob_weighted(0.0)
{
    m_count[0] = 0;
    m_count[1] = 0;
    m_child[0] = NULL;
    m_child[1] = NULL;
}


/* Krichevski-Trofimov estimated log probability accessor */
weight_t CTNode::logProbEstimated(void) const {
    return m_log_prob_est;
}


/* logarithmic weighted probability estimate accessor */
weight_t CTNode::logProbWeighted(void) const {
    return m_log_prob_weighted;
}


/* child corresponding to a particular symbol */
const CTNode *CTNode::child(symbol_t sym) const {
    return m_child[sym];
}


/* the number of times this context been visited */
count_t CTNode::visits(void) const {
    return m_count[Off] + m_count[On];
}


/* compute the logarithm of the KT-estimator update multiplier */
double CTNode::logKTMul(symbol_t sym) const {

    // attempt to hit cache
    if (log_kt_mul_cache_initialized && visits() < LogKTMulCacheSize) {
        return log_kt_mul_cache[m_count[sym]][visits()];
    }

    double kt_mul_numer = double(m_count[sym]) + 0.5;
    double kt_mul_denom = double(visits() + 1);

    return std::log(kt_mul_numer / kt_mul_denom);
}


/* number of descendents of a node in the context tree */
size_t CTNode::size(void) const {

    size_t rval = 1;

    rval += child(Off) ? child(Off)->size() : 0;
    rval += child(On)  ? child(On)->size()  : 0;

    return rval;
}


/* create (if necessary) all of the nodes in the current context */
void ContextTree::createNodesInCurrentContext(const context_t &context) {

    CTNode **ctn = &m_root;

    for (size_t i = 0; i < context.size(); i++) {
        ctn = &((*ctn)->m_child[context[i]]);
        if (*ctn == NULL) {
            void *p = m_ctnode_pool.malloc();
            assert(p != NULL);  // TODO: make more robust
            *ctn = new (p) CTNode();
        }
    }
}


/* updates the context tree with a list of symbols */
void ContextTree::update(const symbol_list_t &symlist) {

    symbol_list_t::const_iterator it = symlist.begin();
    for (; it  != symlist.end(); ++it) {
        update(*it);
    }
}


/* create a context tree of specified maximum depth and size */
ContextTree::ContextTree(size_t depth) :
    m_ctnode_pool(sizeof(CTNode)),
    m_root(new (m_ctnode_pool.malloc()) CTNode()),
    m_depth(depth)
{
    if (!log_kt_mul_cache_initialized) precomputeLogKTMulCache();
}


/* create an empty context tree, used only by Boost Serialization library */
ContextTree::ContextTree(void) :
    m_ctnode_pool(sizeof(CTNode))
{
    if (!log_kt_mul_cache_initialized) precomputeLogKTMulCache();
}


/* copy contructor that performs a deep copy */
ContextTree::ContextTree(const ContextTree &rhs) :
    m_ctnode_pool(sizeof(CTNode)),
    m_history(rhs.m_history),
    m_root(copyCT(rhs.m_root)),
    m_depth(rhs.m_depth)
{
    if (!log_kt_mul_cache_initialized) precomputeLogKTMulCache();
}


/* assignment operator that performs a deep copy */
const ContextTree &ContextTree::operator=(const ContextTree &rhs) {

    if (this != boost::addressof(rhs)) {
        m_ctnode_pool.release_memory();
        deleteCT(m_root);
        m_history     = rhs.m_history;
        m_root        = copyCT(rhs.m_root);
        m_depth       = rhs.m_depth;
    }

    return *this;
}


/* delete the context tree */
ContextTree::~ContextTree(void) {
    deleteCT(m_root);
}


/* recursively copies the nodes in a context tree */
CTNode *ContextTree::copyCT(CTNode *n) {

    if (n == NULL) return NULL;

    CTNode *rval = static_cast<CTNode *>(m_ctnode_pool.malloc());
    assert(rval != NULL);

    *rval = *n;
    rval->m_child[0] = copyCT(n->m_child[0]);
    rval->m_child[1] = copyCT(n->m_child[1]);

    return rval;
}


/* recursively deletes the nodes in a context tree */
void ContextTree::deleteCT(CTNode *n) {

    if (n == NULL) return;

    if (n->m_child[0] != NULL) deleteCT(n->m_child[0]);
    if (n->m_child[1] != NULL) deleteCT(n->m_child[1]);

    m_ctnode_pool.free(n);
}


/* updates the history statistics, without touching the context tree */
void ContextTree::updateHistory(const symbol_list_t &symlist) {

    for (size_t i=0; i < symlist.size(); i++) {
        m_history.push_back(symlist[i]);
    }
}


/* reports the most frequently occuring symbol */
symbol_t ContextTree::mostFrequentSym(void) const {

    return m_root->m_count[On] > m_root->m_count[Off] ? On : Off;
}


/* updates the context tree with a single symbol */
void ContextTree::update(symbol_t sym) {

    // compute the current context
    context_t context; context.reserve(m_depth); getContext(context);

    // if we have not seen enough context, append the symbol
    // to the history buffer and skip updating the context tree
    if (context.size() < m_depth) {
        m_history.push_back(sym);
        return;
    }

    // 1. create new nodes in the context tree (if necessary)
    createNodesInCurrentContext(context);

    // 2. walk down the tree to the relevant leaf context, saving the path as we go
    std::stack<CTNode *, std::vector<CTNode *> > path;
    path.push(m_root); // add the empty context

    // add the path to the leaf nodes
    CTNode *ctn = m_root;
    for (size_t i = 0; i < context.size(); i++) {
        ctn = ctn->m_child[context[i]];
        path.push(ctn);
    }

    // 3. update the probability estimates from the leaf node back up to the root
    for (; !path.empty(); path.pop()) {

        CTNode *n = path.top();

        // update the KT estimate and counts
        double log_kt_mul = n->logKTMul(sym);
        n->m_log_prob_est += log_kt_mul;
        n->m_count[sym]++;

        // update the weighted probabilities
        if (path.size() == m_depth + 1) {
            n->m_log_prob_weighted = n->logProbEstimated();
        } else {
            // computes P_w = log{0.5 * [P_kt + P_w0*P_w1]}
            double log_prob_on  = n->child(On)  ? n->child(On)->logProbWeighted() : 0.0;
            double log_prob_off = n->child(Off) ? n->child(Off)->logProbWeighted() : 0.0;
            double log_one_plus_exp = log_prob_off + log_prob_on - n->logProbEstimated();

            // NOTE: no need to compute the log(1+e^x) if x is large, plus it avoids overflows
            if (log_one_plus_exp < 100.0) log_one_plus_exp = std::log(1.0 + std::exp(log_one_plus_exp));

            n->m_log_prob_weighted = log_point_five + n->logProbEstimated() + log_one_plus_exp;
        }
    }

    // 4. save the new symbol to the context buffer
    m_history.push_back(sym);
}


/* removes the most recently observed symbol from the context tree */
void ContextTree::revert(void) {

    if (m_history.size() == 0) return;

    // 1. remove the most recent symbol from the context buffer
    symbol_t sym = m_history.back();
    m_history.pop_back();

    // compute the current context
    context_t context; context.reserve(m_depth); getContext(context);

    // no need to undo a context tree update if there was
    // not enough context to begin with
    if (context.size() < m_depth) return;

    // 2. determine the path to the leaf nodes
    std::stack<CTNode *, std::vector<CTNode *> > path;
    path.push(m_root); // add the empty context

    // add the path to the leaf nodes
    CTNode *ctn = m_root;
    for (size_t i = 0; i < context.size(); i++) {
        ctn = ctn->m_child[context[i]];
        path.push(ctn);
    }

    // 3. update the probability estimates from the leaf node back up to the root,
    //    deleting any superfluous nodes as we go
    for (; !path.empty(); path.pop()) {

        CTNode *ctn = path.top();

        // undo the previous KT estimate update
        ctn->m_count[sym]--;
        double log_kt_mul = ctn->logKTMul(sym);
        ctn->m_log_prob_est -= log_kt_mul;

        // reclaim memory for any children nodes that now have seen no data
        for (size_t i = 0; i < 2; i++) {
            symbol_t sym = symbols[i];
            if (ctn->m_child[sym] && ctn->m_child[sym]->visits() == 0) {
                m_ctnode_pool.free(ctn->m_child[sym]);
                ctn->m_child[sym] = NULL;
            }
        }

        // update the weighted probabilities
        if (path.size() == m_depth + 1) {
            ctn->m_log_prob_weighted = ctn->logProbEstimated();
        } else {
            // computes P_w = log{0.5 * [P_kt + P_w0*P_w1]}
            double log_prob_on  = ctn->child(On)  ? ctn->child(On)->logProbWeighted() : 0.0;
            double log_prob_off = ctn->child(Off) ? ctn->child(Off)->logProbWeighted() : 0.0;
            double log_one_plus_exp = log_prob_off + log_prob_on - ctn->logProbEstimated();

            // NOTE: no need to compute the log(1+e^x) if x is large, plus it avoids overflows
            if (log_one_plus_exp < 100.0) log_one_plus_exp = std::log(1.0 + std::exp(log_one_plus_exp));

            ctn->m_log_prob_weighted = log_point_five + ctn->logProbEstimated() + log_one_plus_exp;
        }
    }
}


/* gives the estimated probability of observing a particular symbol */
double ContextTree::predict(symbol_t sym) {

    // if we haven't sufficient context to make an informed
    // prediction then guess uniformly randomly
    if (m_history.size() + 1 <= m_depth) return 0.5;

    // prob(sym | history) = prob(sym and history) / prob(history)
    double log_prob_history = m_root->logProbWeighted();
    update(sym);
    double log_prob_sym_and_history = m_root->logProbWeighted();
    revert();

    return std::exp(log_prob_sym_and_history - log_prob_history);
}


/* gives the estimated probability of observing a particular sequence */
double ContextTree::predict(symbol_list_t symlist) {

    // if we haven't enough context to make an informed
    // prediction then guess uniformly randomly
    if (m_history.size() + symlist.size() <= m_depth) {
        double exp = -double(symlist.size());
        return std::pow(2.0, exp);
    }

    // prob(sym1 ^ sym2 ^ ... | history) = prob(sym1 ^ sym2 ^ ... and history) / prob(history)
    double log_prob_history = logBlockProbability();
    update(symlist);
    double log_prob_syms_and_history = logBlockProbability();

    symbol_list_t::const_iterator it = symlist.begin();
    for (; it != symlist.end(); ++it) {
        revert();
    }

    return std::exp(log_prob_syms_and_history - log_prob_history);
}


/* clear the entire context tree */
void ContextTree::clear(void) {

    m_history.clear();
    deleteCT(m_root);
    m_root = new (m_ctnode_pool.malloc()) CTNode();
}


/* generate a specified number of random symbols
   distributed according to the context tree statistics */
void ContextTree::genRandomSymbols(randgen_t &rng, symbol_list_t &symbols, size_t bits) {

    genRandomSymbolsAndUpdate(rng, symbols, bits);

    // restore the context tree to it's original state
    for (size_t i=0; i < bits; i++) revert();
}


/* generate a specified number of random symbols distributed according to the context tree
   statistics and update the context tree with the newly generated bits */
void ContextTree::genRandomSymbolsAndUpdate(randgen_t &rng, symbol_list_t &symbols, size_t bits) {

    symbols.clear();

    for (size_t i=0; i < bits; i++) {
        // flip a biased coin for each bit
        symbol_t rand_sym = rng() < predict(Off) ? Off : On;
        symbols.push_back(rand_sym);
        update(rand_sym); // TODO: optimise this loop
    }
}


/* the depth of the context tree */
size_t ContextTree::depth(void) const {

    return m_depth;
}


/* the size of the stored history */
size_t ContextTree::historySize(void) const {

    return m_history.size();
}


/* get the n'th most recent history symbol, NULL if doesn't exist */
const symbol_t *ContextTree::nthHistorySymbol(size_t n) const {

    return n < m_history.size() ? &m_history[n] : NULL;
}


/* sets the function that computes the context */
void ContextTree::setContextFunctor(boost::function<void (context_t&)> functor) {

    m_context_functor = functor;
}


/* shrinks the history down to a former size */
void ContextTree::revertHistory(size_t newsize) {

    assert(newsize <= m_history.size());
    while (m_history.size() > newsize) m_history.pop_back();
}


/* number of nodes in the context tree */
size_t ContextTree::size(void) const {

    return m_root->size();
}


/* the logarithm of the block probability of the whole sequence */
double ContextTree::logBlockProbability(void) {
    return m_root->logProbWeighted();
}


/* create a new factored context tree */
FactoredContextTree::FactoredContextTree(size_t num_factors, size_t depth) {

    for (size_t i=0; i < num_factors; i++) {
        m_cts.push_back(new ContextTree(depth));
    }
}


/* copy contructor performs a deep copy */
FactoredContextTree::FactoredContextTree(const FactoredContextTree &rhs) {

    m_cts     = rhs.m_cts;
}


/* assignment operator performs a deep copy */
const FactoredContextTree &FactoredContextTree::operator=(const FactoredContextTree &rhs) {

    if (this != boost::addressof(rhs)) {
        m_cts.clear();
        m_cts     = rhs.m_cts;
    }

    return *this;
}


/* updates the factored context tree with a new block of binary symbols.
   each factor gets updated exactly once. */
void FactoredContextTree::update(const symbol_list_t &symlist) {

    assert(symlist.size() == m_cts.size());

    size_t c = 0;
    symbol_list_t::const_iterator it = symlist.begin();
    for (; it  != symlist.end(); ++it) {
        update(c++, *it);
    }
}


/* updates the history with a new list of binary symbols */
void FactoredContextTree::updateHistory(const symbol_list_t &symlist) {

    for (size_t i=0; i < m_cts.size(); i++) {
        m_cts[i].updateHistory(symlist);
    }
}


/* removes the most recently observed symbol from the context tree */
void FactoredContextTree::revert(size_t offset) {

    m_cts[offset].revert();
    for (size_t i=0; i < m_cts.size(); i++) {
        if (i != offset) m_cts[i].m_history.pop_back();
    }
}


/* shrinks the history down to a former size */
void FactoredContextTree::revertHistory(size_t newsize) {

    for (size_t i=0; i < m_cts.size(); i++) {
        m_cts[i].revertHistory(newsize);
    }
}


/* gives the estimated probability of observing a particular sequence of symbols,
   given our current history */
double FactoredContextTree::predict(symbol_list_t symlist) {

    if (symlist.size() == 0) return 1.0;

    // if we haven't enough context to make an informed
    // prediction then guess uniformly randomly
    if (historySize() + symlist.size() <= m_cts[0].depth()) {
        double exp = -double(symlist.size());
        return std::pow(2.0, exp);
    }

    // prob(sym1 ^ sym2 ^ ... | history) = prob(sym1 ^ sym2 ^ ... and history) / prob(history)
    double log_prob_history = logBlockProbability();
    update(symlist);
    double log_prob_syms_and_history = logBlockProbability();

    // restore the context tree to it's original state
    for (int i=int(symlist.size())-1; i >= 0; i--) revert(i);

    return std::exp(log_prob_syms_and_history - log_prob_history);
}


/* generate a specified number of random symbols distributed according to the
   factored context tree statistics */
void FactoredContextTree::genRandomSymbols(randgen_t &rng, symbol_list_t &symbols, size_t bits) {

    genRandomSymbolsAndUpdate(rng, symbols, bits);

    // restore the context tree to it's original state
    for (int i=int(bits)-1; i >= 0; i--) revert(i);
}


/* generate a specified number of random symbols distributed according to
   the context tree statistics, updating our model with the generated bits */
void FactoredContextTree::genRandomSymbolsAndUpdate(randgen_t &rng, symbol_list_t &symbols, size_t bits) {

    assert(bits == m_cts.size());

    symbols.clear();
    if (bits == 0) return;

    for (size_t i=0; i < bits; i++) {

        // if we haven't enough context then guess uniformly at random
        if (historySize() + symbols.size() <= m_cts[i].depth()) {

            symbol_t sym = rng() < 0.5 ? Off : On;
            symbols.push_back(sym);
            update(i, sym);

        } else {

            double log_prob_history = logBlockProbability();

            // take a guess! if we guess right, we save effort.
            symbol_t guess = m_cts[i].mostFrequentSym();
            update(i, guess);
            double log_prob_sym_and_history = logBlockProbability();
            double p = std::exp(log_prob_sym_and_history - log_prob_history);
            symbols.push_back(guess);

            // if our guess is wrong, rollback and redo
            if (rng() >= p) {
                symbol_t not_guess = flip(guess);
                revert(i);
                update(i, not_guess);
                symbols[symbols.size()-1] = not_guess;
            }
        }
    }
}


/* clear the entire factored context tree */
void FactoredContextTree::clear(void) {

    for (size_t i=0; i < m_cts.size(); i++) {
        m_cts[i].clear();
    }
}

/*  the depth of the factored context tree */
size_t FactoredContextTree::depth(void) const {

    return m_cts[0].depth();
}


/* the size of the stored history */
size_t FactoredContextTree::historySize(void) const {

    return m_cts[0].historySize();
}


/* number of nodes in the context tree */
size_t FactoredContextTree::size(void) const {

    size_t total = 0;

    for (size_t i=0; i < m_cts.size(); i++) {
        total += m_cts[i].size();
    }

    return total;
}


/* get the n'th history symbol, NULL if doesn't exist */
const symbol_t *FactoredContextTree::nthHistorySymbol(size_t n) const {

    return m_cts[0].nthHistorySymbol(n);
}


/* sets the function that computes the context */
void FactoredContextTree::setContextFunctor(boost::function<void (context_t&)> functor) {

    for (size_t i=0; i < m_cts.size(); i++) {
        m_cts[i].setContextFunctor(functor);
    }
}


/* update the factored context tree with one symbol */
void FactoredContextTree::update(size_t offset, symbol_t sym) {

    assert(offset < m_cts.size());

    m_cts[offset].update(sym);

    // update histories of other context trees
    for (size_t i=0; i < m_cts.size(); i++) {
        if (i != offset) m_cts[i].m_history.push_back(sym);
    }
}


/* determine the probability of the next symbol */
double FactoredContextTree::predict(size_t offset, symbol_t sym) {

    assert(offset < m_cts.size());

    return m_cts[offset].predict(sym);
}


/* the logarithm of the block probability of the whole sequence */
double FactoredContextTree::logBlockProbability(void) {

    double total = 0.0;

    for (size_t i=0; i < m_cts.size(); i++) {
        total += m_cts[i].logBlockProbability();
    }

    return total;
}

