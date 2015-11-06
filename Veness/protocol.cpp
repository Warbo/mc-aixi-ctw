#include "protocol.hpp"

#include <string>
#include <sstream>

// boost includes
#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>
#include <boost/filesystem.hpp>
#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>

#include "pipsqueak.hpp"
#include "agent.hpp"
#include "log.hpp"
#include "search.hpp"


// our hive of agents
static std::auto_ptr<Hive> hive;


/* gets the agent by identification number, NULL on failure */
Agent *getAgent(size_t id) {

    return (hive.get() == NULL) ? NULL : (*hive)[id];
}


// internal command interface
struct Command;

// finds the command record pertaining to a given command name, NULL if not found
static const Command *findCommand(const std::string &cmd);

// how many commands are there?
static size_t numCommands(void);

// gets the i'th command, NULL if command does not exist
static const Command *getCommand(size_t i);


// describes the internal command information record
struct Command {

    // name of the command
    virtual const char *name(void) const = 0;

    // help message
    virtual const char *help(void) const = 0;

    // run the command
    virtual bool execute(Agent &agent, const std::string &input) const = 0;
};


// resets the state of an agent to what it would be having no history
struct CmdReset : public Command {

    bool execute(Agent &agent, const std::string &input) const {

        for (size_t i=0; i < hive->count(); i++) {
            (*hive)[i]->reset();
        }

        LINFO_ << "reset command: agent reset.";

        return true;
    }

    const char *name(void) const {
        return "reset";
    }

    const char *help(void) const {
        return "reset the state of an agent";
    }
};


// saves the agent to disk. default is taken from the command line.
struct CmdSave : public Command {

    bool execute(Agent &agent, const std::string &input) const {

        // only the primary agent can be saved
        if (agent.id() != 0) return false;

        std::string filename = input;

        if (filename == "" && options.count("agent-save") > 0) {
            filename = options["agent-save"].as<std::string>();
        }

        return filename != "" ? agent.save(filename) : false;
    }

    const char *name(void) const {
        return "save";
    }

    const char *help(void) const {
        return "save [filename] - saves agent to disk";
    }
};


// loads an agent from disk
struct CmdLoad : public Command {

    bool execute(Agent &agent, const std::string &input) const {

        std::string filename = input;

        if (filename == "" && options.count("agent-load") > 0) {
            filename = options["agent-load"].as<std::string>();
        }

        if (!boost::filesystem::exists(filename)) {
            LERR_ << "load failed as " << filename << " does not exist.";
            return false;
        }

        unsigned int threads = options["threads"].as<unsigned int>();
        hive.reset(new Hive(threads, options["agent-load"].as<std::string>()));
        return true;
    }

    const char *name(void) const {
        return "load";
    }

    const char *help(void) const {
        return "load [filename] - loads an agent from disk";
    }
};


// terminates the agent <-> environment loop
struct CmdQuit : public Command {

    bool execute(Agent &agent, const std::string &) const {

        std::string filename = "";
        if (options.count("agent-save") > 0) {
            filename = options["agent-save"].as<std::string>();
        }

        if (filename != "") {
            bool rval = agent.save(filename);
            if (!rval) LERR_ << "failed to save agent to output file: " << filename;
            std::exit(1);
        }

        std::exit(0);
    }

    const char *name(void) const {
        return "quit";
    }

    const char *help(void) const {
        return "terminates the agent <-> environment loop";
    }
};


// displays the age of the agent
struct CmdAge : public Command {

    bool execute(Agent &agent, const std::string &input) const {

        std::cout << agent.age() << std::endl;
        return true;
    }

    const char *name(void) const {
        return "age";
    }

    const char *help(void) const {
        return "show the age of the agent in life cycles";
    }
};


// displays the look-ahead amount the agent uses
struct CmdHorizon : public Command {

    bool execute(Agent &agent, const std::string &input) const {

        std::cout << agent.horizon() << std::endl;
        return true;
    }

    const char *name(void) const {
        return "horizon";
    }

    const char *help(void) const {
        return "show the search horizon length";
    }
};



// displays a short message on how to use the system
struct CmdHelp : public Command {

    bool execute(Agent &agent, const std::string &input) const {

        for (size_t i=0; i < numCommands(); i++) {

            const Command *cmd = getCommand(i);
            assert(cmd != NULL);

            std::cout << cmd->name() << "\t\t" << cmd->help() << std::endl;
        }
        return true;
    }

    const char *name(void) const {
        return "help";
    }

    const char *help(void) const {
        return "show a short usage message";
    }
};


// command listing
static const Command *commands[] = {
    new CmdHelp(),
    new CmdQuit(),
    new CmdLoad(),
    new CmdReset(),
    new CmdSave(),
    new CmdAge(),
    new CmdHorizon()
};


/* finds the command record pertaining to a given command name, NULL if not found */
static const Command *findCommand(const std::string &cmd) {

    for (size_t i=0; i < numCommands(); i++) {
        if (cmd == commands[i]->name()) return commands[i];
    }

    return NULL;
}


/* gets the i'th command, NULL if command does not exist. */
static const Command *getCommand(size_t i) {

    return i < numCommands() ? commands[i] : NULL;
}


/* how many commands are there? */
static size_t numCommands(void) {

    return sizeof(commands) / sizeof(commands[0]);
}


/* executes a user command */
bool dispatchCommand(Agent &agent, const std::string &command) {

    typedef boost::tokenizer<boost::char_separator<char> > Tok;
    boost::char_separator<char> sep(" ");
    Tok tok(command, sep);

    std::string first = *tok.begin();

    // command must begin with a colon
    if (first[0] != ':') return false;

    LINFO_ << "received " << command;

    Tok::iterator it = tok.begin();
    std::string args = "";
    if (++it != tok.end()) args = *it;

    const Command *cmd = findCommand(first.substr(1));
    if (cmd != NULL) {
        bool rval = cmd->execute(agent, args);
        if (!rval) LERR_ << "command: " << command << " failed.";
    }

    return cmd != NULL;
}


/* agent <-> environment main-loop */
void mainLoop(std::istream &in, std::ostream &out) {

    // random number generator
    static randsrc_t rng;
    static randgen_t urng(rng);

    std::string buf;
    symbol_list_t percept_syms;

    // create the agent hive, one agent per thread
    unsigned int threads = options["threads"].as<unsigned int>();
    if (options.count("agent-load") > 0) {
        hive.reset(new Hive(threads, options["agent-load"].as<std::string>()));
    } else {
        hive.reset(new Hive(threads));
    }
    Agent &ai = *(*hive)[0];

    // determine the exploration options
    bool explore = options.count("exploration") > 0;
    double explore_rate, explore_decay;
    if (explore) {
        explore_rate  = options["exploration"].as<double>();
        explore_decay = options["explore-decay"].as<double>();
    }

    // determine the terminate age, if any
    bool terminate_check = options.count("terminate-age") > 0;
    age_t terminate_age;
    if (terminate_check) {
        terminate_age = ai.age() + options["terminate-age"].as<age_t>();
    }

    // record initial session statistics
    age_t start_age = ai.age();
    reward_t start_rew = ai.reward();
    double avg_session_percept_prob = 0.0;

    while (std::getline(in, buf)) {

        // handle user commands
        if (buf.length() > 0 && buf[0] == ':') {
            if (!dispatchCommand(ai, buf)) {
                LERR_ << "could not find command: " << buf;
            }
            continue;
        }

        // check for agent termination
        if (terminate_check && ai.age() > terminate_age) {
            LINFO_ << "terminating agent...";
            break;
        }

        // gather a percept from the environment -> agent channel
        if (!ai.percept2symlist(buf, percept_syms)) throw BadPerceptException();

        // keep track of our predictive accuracy
        double percept_prob = ai.perceptProbability(percept_syms);
        double n = double(ai.age() - start_age);
        avg_session_percept_prob = (percept_prob + n * avg_session_percept_prob) / (n + 1.0);

        LINFO_ << "percept prob: " << percept_prob << ", "
               << "avg session percerpt prob: " << avg_session_percept_prob;

        // record some information from the current cycle
        reward_t r = ai.rewardFromPercept(percept_syms);
        LINFO_ << "Received reward of " << r << " at timestep " << ai.age() << std::endl;

        // update the hive's internal model with the new percept
        hive->modelUpdate(percept_syms);

        // determine the best exploitive action, or explore
        action_t action;
        if (explore) LINFO_ << "Explorating with probability: " << explore_rate;
        if (explore && urng() < explore_rate) {
            LINFO_ << "Exploring!";
            action = ai.selectRandomAction(urng);
        } else {
            action = search(*hive);
        }

        LINFO_ << "Selected action: " << action << " from {0,"
               << (ai.numActions() - 1) << "}" << std::endl;

        // send an action across the agent -> environment channel
        ai.sendAction(action, out);

        // update the agent's internal model with the action it performed
        hive->modelUpdate(action);

        // update the exploration rate
        if (explore) explore_rate *= explore_decay;

        ai.logDynamicProperties();
    }

    // display session summary
    std::ostringstream oss;
    oss << "Session summary: " << (ai.reward() - start_rew)
        << " reward from " << (ai.age() - start_age - 1) << " cycles.";
    LINFO_ << oss.str();
    out << oss.str();
}

