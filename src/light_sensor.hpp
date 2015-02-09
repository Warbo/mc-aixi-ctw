#ifndef __LIGHTSENSOR_HPP__
#define __LIGHTSENSOR_HPP__

#include "environment.hpp"

/** The agent's actions control an LED and the reward is given by the value from
 * a light-sensitive resistor. These are accessed remotely via an Arduino.
 *
 * Domain characteristics:
 * - environment: "light-sensor"
 * - maximum action: 1 (1 bit)
 * - maximum observation: 1024 (11 bits)
 * - maximum reward: 1024 (11 bits)
 */
class LightSensor : public Environment {
public:

	// set up the initial environment percept
	LightSensor(options_t &options);

	virtual void performAction(const action_t action);

	virtual action_t maxAction() const { return 1; }
	virtual percept_t maxObservation() const { return 9; }
	virtual percept_t maxReward() const { return maxObservation(); }

	virtual std::string print() const;

private:
	virtual char* read_json();
	virtual char read_serial();
	virtual char* find_value(char*);
	virtual int read_value(char*);

	/** Action: agent turns on its LED. */
	static const action_t turnOff;

	/** Action: agent turns off its LED. */
	static const action_t turnOn;

	/** File descriptors for serial access. */
	int infd;
	int outfd;
	int step;
        int off_val;
};

#endif // __LIGHTSENSOR_HPP__
