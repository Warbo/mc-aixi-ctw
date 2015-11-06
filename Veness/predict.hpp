#ifndef __PREDICT_HPP__
#define __PREDICT_HPP__

#include <vector>
#include <deque>

// boost includes
#include <boost/random.hpp>
#include <boost/pool/pool.hpp>
#include <boost/serialization/deque.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/function.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/ptr_container/serialize_ptr_vector.hpp>

#include "pipsqueak.hpp"


// stores symbol occurrence counts
typedef uint32_t count_t;

// holds context weights
typedef double weight_t;

// stores the agent's history in terms of primitive symbols
typedef std::deque<symbol_t> history_t;


// context tree node
class CTNode {

    friend class boost::serialization::access;

    friend class ContextTree;

    public:
        /// log weighted blocked probability
        weight_t logProbWeighted(void) const;

        /// log KT estimated probability
        weight_t logProbEstimated(void) const;

        /// child corresponding to a particular symbol
        const CTNode *child(symbol_t sym) const;

        /// the number of times this context been visited
        count_t visits(void) const;

        /// number of descendents
        size_t size(void) const;

        /// serialization routine
        template<class Archive>
        void serialize(Archive &ar, const unsigned int version);

    private:

        CTNode(void);

        // compute the logarithm of the KT-estimator update multiplier
        double logKTMul(symbol_t sym) const;

        weight_t m_log_prob_est;
        weight_t m_log_prob_weighted;

        // one slot for each symbol
        count_t m_count[2];  // a,b in CTW literature
        CTNode *m_child[2];
};


/* serialization routine - needs to be accessible from header */
template<class Archive>
inline void CTNode::serialize(Archive &ar, const unsigned int version) {

    ar & m_log_prob_est;
    ar & m_log_prob_weighted;
    ar & m_count;
    ar & m_child;
}



// a context tree for binary data
class ContextTree {

    friend class boost::serialization::access;
    friend class FactoredContextTree;

    public:

        /// create a context tree of specified maximum depth and size
        ContextTree(size_t depth);

        /// copy contructor / assignment operator performs a deep copy
        explicit ContextTree(const ContextTree &rhs);
        const ContextTree &operator=(const ContextTree &rhs);

        /// delete the context tree
        ~ContextTree(void);

        /// updates the context tree with a new binary symbol
        void update(symbol_t sym);
        void update(const symbol_list_t &symlist);
        void updateHistory(const symbol_list_t &symlist);

        /// removes the most recently observed symbol from the context tree
        void revert(void);

        /// shrinks the history down to a former size
        void revertHistory(size_t newsize);

        /// gives the estimated probability of observing
        /// a particular symbol or sequence
        double predict(symbol_t sym);
        double predict(symbol_list_t symlist);

        /// reports the most frequently occuring symbol
        symbol_t mostFrequentSym(void) const;

        /// generate a specified number of random symbols
        /// distributed according to the context tree statistics
        void genRandomSymbols(randgen_t &rng, symbol_list_t &symbols, size_t bits);

        /// generate a specified number of random symbols distributed according to the context tree
        /// statistics and update the context tree with the newly generated bits
        void genRandomSymbolsAndUpdate(randgen_t &rng, symbol_list_t &symbols, size_t bits);

        /// clear the entire context tree
        void clear(void);

        /// the depth of the context tree
        size_t depth(void) const;

        /// the size of the stored history
        size_t historySize(void) const;

        /// number of nodes in the context tree
        size_t size(void) const;

        /// get the n'th history symbol, NULL if doesn't exist
        const symbol_t *nthHistorySymbol(size_t n) const;

        /// serialization routine
        template<class Archive>
        void serialize(Archive &ar, const unsigned int version);

        /// sets the function that computes the context
        void setContextFunctor(boost::function<void (context_t&)> functor);

        // the logarithm of the block probability of the whole sequence
        double logBlockProbability(void);

    private:

        // only used by Boost Serialization
        ContextTree(void);

        // compute the current context
        void getContext(context_t &context) const;

        // create (if necessary) all of the nodes in the current context
        void createNodesInCurrentContext(const context_t &context);

        // recursively deletes the nodes in a context tree
        void deleteCT(CTNode *root);

        // recursively copies the nodes in a context tree
        CTNode *copyCT(CTNode *root);

        // not serialized
        boost::pool<> m_ctnode_pool;
        boost::function<void (context_t&)> m_context_functor;

        history_t m_history;
        CTNode *m_root;
        size_t m_depth;
};


/* serialization routine - needs to be accessible from header */
template<class Archive>
inline void ContextTree::serialize(Archive &ar, const unsigned int version) {

    ar & m_history;
    ar & m_root;
    ar & m_depth;
}


// stores a factored context tree
class FactoredContextTree {

    friend class boost::serialization::access;

    public:

        /// create a factored context tree
        FactoredContextTree(size_t num_factors, size_t depth);

        /// copy contructor / assignment operator performs a deep copy
        explicit FactoredContextTree(const FactoredContextTree &rhs);
        const FactoredContextTree &operator=(const FactoredContextTree &rhs);

        /// serialization routine
        template<class Archive>
        void serialize(Archive &ar, const unsigned int version);

        /// updates the factored context tree with a new binary symbol
        void update(const symbol_list_t &symlist);
        void updateHistory(const symbol_list_t &symlist);

        /// shrinks the history down to a former size
        void revertHistory(size_t newsize);

        /// gives the estimated probability of observing
        /// a particular symbol or sequence
        double predict(symbol_list_t symlist);

        /// generate a specified number of random symbols
        /// distributed according to the context tree statistics
        void genRandomSymbols(randgen_t &rng, symbol_list_t &symbols, size_t bits);

        /// generate a specified number of random symbols distributed according to
        /// the context tree statistics, updating our model with the generated bits
        void genRandomSymbolsAndUpdate(randgen_t &rng, symbol_list_t &symbols, size_t bits);

        /// clear the entire context tree
        void clear(void);

        /// the depth of the context tree
        size_t depth(void) const;

        /// the size of the stored history
        size_t historySize(void) const;

        /// number of nodes in the context tree
        size_t size(void) const;

        /// get the n'th history symbol, NULL if doesn't exist
        const symbol_t *nthHistorySymbol(size_t n) const;

        /// sets the function that computes the context
        void setContextFunctor(boost::function<void (context_t&)> functor);

        // the logarithm of the block probability of the whole sequence
        double logBlockProbability(void);

        // removes the most recently observed symbol from the context tree
        void revert(size_t offset);

    private:

        // only used by Boost Serialization
        FactoredContextTree(void) { }

        // update a single context tree factor
        void update(size_t offset, symbol_t sym);

        // make a prediction from a single context tree factor
        double predict(size_t offset, symbol_t sym);

        boost::ptr_vector<ContextTree> m_cts;
};


/* serialization routine - needs to be accessible from header */
template<class Archive>
inline void FactoredContextTree::serialize(Archive &ar, const unsigned int version) {
    ar & m_cts;
}

#endif // __PREDICT_HPP__

