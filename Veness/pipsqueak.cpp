#include "pipsqueak.hpp"

#include <iostream>
#include <limits>
#include <exception>
#include <stdexcept>

// boost includes
#include <boost/program_options.hpp>

#include "protocol.hpp"
#include "log.hpp"

namespace po = boost::program_options;


// program details
static const char program_name[] = "mc-aixi(fac-ctw)";
static const char version[]      = "1.0";
static const char authors[]      = "Joel Veness";


// stores the application configuration options and their description
po::variables_map options;
static po::options_description options_desc("Usage", 100);


/* load program options into a variable map */
static void initOptions(int argc, char *argv[], po::variables_map &vm) {

    options_desc.add_options()

        ("help", "")

        ("ct-depth",
         po::value<unsigned int>()->default_value(3),
         "maximum depth of the context tree used for prediction")

        ("reward-bits",
         po::value<unsigned int>()->default_value(1),
         "how many bits are used to encode the reward signal")

        ("observation-bits",
         po::value<unsigned int>()->default_value(1),
         "how many bits are used to encode the observation signal")

        ("cycle-length-ms",
         po::value<unsigned int>(),
         "milliseconds after receiving a percept to choose an action")

        ("agent-horizon",
         po::value<unsigned int>()->default_value(16),
         "the number of percept/action pairs to look forward")

        ("agent-actions",
         po::value<unsigned int>()->default_value(4),
         "the number of distinct actions the agent can do")

        ("agent-load",
         po::value<std::string>(),
         "load a pre-existing agent from a file")

        ("reward-encoding",
         po::value<std::string>()->default_value("base2"),
         "how the agent interprets the reward encoding (bitcount/base2)")

        ("agent-save",
         po::value<std::string>(),
         "save the context tree to file upon exit")

        ("controller",
         po::value<std::string>()->default_value("mcts"),
         "control algorithm to use: (mcts/mc/random)")

        ("threads",
         po::value<unsigned int>()->default_value(1),
         "number of search threads to use?")

        ("exploration",
         po::value<double>(),
         "probability of playing a random move")

        ("explore-decay",
         po::value<double>()->default_value(1.0),
         "a value between [0.0,1.0] that defines the geometric decay of the exploration rate")

        ("bootstrapped-playouts",
         "use a self-improving playout policy")

        ("terminate-age",
         po::value<age_t>(),
         "how many agent/environment cycles before the agent needs to close?")

        ("binary-io",
         "use native (faster but non-portable) binary file i/o")

        ("memsearch",
         po::value<size_t>()->default_value(32),
         "maximum amount of memory used by the search tree in megabytes")

        ("mc-simulations",
         po::value<size_t>(),
         "specify the number of MC simulations per cycle")
    ;

    po::store(po::parse_command_line(argc, argv, options_desc), vm);
    po::notify(vm);
}


/* display the program usage options and quit. */
static void showHelp(void) {

    std::cout << std::endl;
    std::cout << program_name << " " << version << ", by " << authors << std::endl;
    std::cout << "-------------------------------" << std::endl << std::endl;
    std::cout << "An approximate universal artificial intelligence for tree based environments." << std::endl << std::endl;
    std::cout << "The environment communicates percepts to the agent over stdin." << std::endl;
    std::cout << "The agent responds to the environment over stdout." << std::endl << std::endl;
    std::cout << "All communication channels are binary." << std::endl;
    std::cout << std::endl << options_desc << std::endl;
    exit(0);
}


/* checks for illegal combinations of configuration parameters,
   throwing an exception if an illegal combination is detected. */
static void processOptions(const po::variables_map &vm) {

    if (vm.count("help") > 0) {
        showHelp();
    }

    if (vm.count("threads") > 0) {

        unsigned int threads = vm["threads"].as<unsigned int>();

        if (threads < 1) {
            throw std::invalid_argument("# threads must be > 0.");
        }

        if (threads > 32) {
            throw std::invalid_argument("cannot use more than 32 threads.");
        }

        if (threads > 1 && vm["controller"].as<std::string>() != "mcts") {
            throw std::invalid_argument("using more than 1 thread requires controller=mcts");
        }
    }

    if (vm.count("agent-actions") > 0) {
        if (vm["agent-actions"].as<unsigned int>() < 2) {
            throw std::invalid_argument("the number of distinct actions must be at least 2.");
        }
    }

    if (vm.count("agent-horizon") > 0) {
        if (vm["agent-horizon"].as<unsigned int>() < 1) {
            throw std::invalid_argument("the horizon must be at least one.");
        }
    }

    if (vm.count("reward-bits") > 0) {
        if (vm["reward-bits"].as<unsigned int>() < 1) {
            throw std::invalid_argument("the number of reward bits must be positive.");
        }
        if (vm["reward-bits"].as<unsigned int>() > 32) {
            throw std::invalid_argument("the number of reward bits must be 32 or less.");
        }
    }

    if (vm.count("observation-bits") > 0) {
        if (vm["observation-bits"].as<unsigned int>() < 1) {
            throw std::invalid_argument("the number of observation bits must be positive.");
        }
    }

    if (vm.count("reward-encoding") > 0) {
        if (vm["reward-encoding"].as<std::string>() != "base2" &&
            vm["reward-encoding"].as<std::string>() != "bitcount") {
            throw std::invalid_argument("invalid reward encoding.");
        }
    }

    if (vm.count("controller") > 0) {
        if (vm["controller"].as<std::string>() != "mc"   &&
            vm["controller"].as<std::string>() != "mcts" &&
            vm["controller"].as<std::string>() != "random") {
            throw std::invalid_argument("invalid controller.");
        }
    }

    if (vm.count("exploration") > 0) {
        if (vm["controller"].as<std::string>() == "random") {
            throw std::invalid_argument(
                "exploration and controller=random options are incompatible"
            );
        }

        double x = vm["exploration"].as<double>();
        if (x < 0.0 || x > 1.0) {
            throw std::invalid_argument(
                "exploration probability must lie within [0..1]"
            );
        }
    }

    if (vm.count("explore-decay") > 0) {
        double x = vm["explore-decay"].as<double>();
        if (x < 0.0 || x > 1.0) {
            throw std::invalid_argument(
                "exploration decay must lie within [0..1]"
            );
        }
    }

    if (vm.count("mc-simulations") > 0) {
        size_t x = vm["mc-simulations"].as<size_t>();
        if (x == 0) {
            throw std::invalid_argument(
                "# of mc-simulatioins must be > 0"
            );
        }

        if (vm.count("cycle-length-ms") > 0) {
            throw std::invalid_argument(
                "mc-simulations and cycle-length-ms options are incompatible"
            );
        }
    }

    if (vm["controller"].as<std::string>() == "mcts") {
        if (!vm.count("mc-simulations") && !vm.count("cycle-length-ms")) {
            throw std::invalid_argument(
                "either mc-simulations or cycle-length-ms must be specified"
            );
        }
    }
}


/* application entry point */
int main(int argc, char *argv[]) {

    try {

        initLogs();

        // load program configuration
        initOptions(argc, argv, options);

        // check and apply program configuration
        processOptions(options);

        // enter agent <-> environment communication loop
        mainLoop(std::cin, std::cout);

    } catch (std::exception &e) {

        LERR_ << "error: " << e.what() << std::endl;
        return 1;

    } catch (...) {

        LERR_ << "error: unknown exception" << std::endl;
        return 1;
    }

    LINFO_ << "agent terminated";

    return 0;
}

