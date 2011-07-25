#include <cassert>
#include "light_sensor.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <termios.h>
#include <unistd.h>
#include <unistd.h>

LightSensor::LightSensor(options_t &options) {
	char* reply;
	// Input line
	infd = open("/dev/ttyACM0", O_RDWR | O_NOCTTY | O_NDELAY);
	if (infd == -1 )
	{
		perror("open_port: Unable to open /dev/ttyACM0");
	}
	else
	{
		fcntl(infd, F_SETFL,0);
		printf("Connected...\n");
	}
	// Output line
	outfd = open("/dev/ttyACM0", O_RDWR | O_NOCTTY | O_NDELAY);
	if (outfd == -1 )
	{
		perror("open_port: Unable to open /dev/ttyACM0");
	}
	else
	{
		fcntl(outfd, F_SETFL,0);
		printf("Connected...\n");
	}

	// Set the input and output in the Arduino
	// TODO: Check the output for errors
	write(outfd,"{\"query\":\"status\"}",18);
	usleep(100);
	write(outfd,"{\"mode\":{\"pin\":12,\"mode\":\"output\"}}",35);
	reply = read_json();
	if (reply) {
		free(reply);
	}
	write(outfd,"{\"mode\":{\"pin\":1,\"mode\":\"input\"}}",33);
	reply = read_json();
	if (reply) {
		free(reply);
	}

	// Now we calibrate our readings. First we turn the light off:
	write(outfd,"{\"write\":{\"pin\":12,\"type\":\"digital\",\"value\":0}}\r",48);
	int off_val;
	int on_val;
	char* start;
	reply = read_json();
	free(reply);
	// Then we read the light sensor
	write(outfd,"{\"read\":{\"pin\":1,\"type\":\"analogue\"}}",36);
	reply = read_json();
	if (reply) {
		start = find_value(reply);
		if (start) {
			off_val = read_value(start);
		}
		else {
			exit(1);
		}
		free(reply);
		// Then we turn on the light:
		write(outfd,"{\"write\":{\"pin\":12,\"type\":\"digital\",\"value\":1}}\r",48);
		reply = read_json();
		free(reply);
		// Then we read the light sensor
		write(outfd,"{\"read\":{\"pin\":1,\"type\":\"analogue\"}}",36);
		reply = read_json();
		if (reply) {
			start = find_value(reply);
			if (start) {
				on_val = read_value(start);
			}
			else {
				exit(1);
			}
			free(reply);
		}
		else {
			exit(1);
		}
	}
	else {
		exit(1);
	}
	// The threshold between good and bad is half-way between on and off
	threshold = on_val - ((on_val - off_val) / 2);
	
	// Initial percept
	m_observation = 0;
	m_reward = m_observation;
}

void LightSensor::performAction(const action_t action) {
	//assert(isValidAction(action));
	m_action = action;			// Save the action
	int wr;						// Used for writing serial
	char* buffer;				// This stores the JSON we get over serial
	char* value_start;			// This points into our JSON at "value"
	int observation = 0;		// Default to zero in case we get no value
	
	switch (m_action)
	{
		case 0:
			wr=write(outfd,"{\"write\":{\"pin\":12,\"type\":\"digital\",\"value\":0}}\r",48);
			break;
		case 1:
			wr=write(outfd,"{\"write\":{\"pin\":12,\"type\":\"digital\",\"value\":1}}\r",48);
			break;
	}
	// Read from the serial line until we have a whole JSON object
	// This is the reply to our write. It should be {}
	this->read_json();
	wr=write(outfd,"{\"read\":{\"pin\":1,\"type\":\"analogue\"}}",36);
	buffer = this->read_json();

	if (buffer) {
		// Look for "value" in this object
		value_start = find_value(buffer);
		if (value_start) {
			// Read whatever number comes afterwards
			observation = read_value(value_start);
		}
		free(buffer);
	}

	if (observation < threshold) {
		m_observation = 0;
		m_reward = 0;
	}
	else {
		m_observation = 1;
		m_reward = 1;
	}
}

std::string LightSensor::print() const {
	std::ostringstream out;
	out << "prediction: " << (m_action)
		<< ", observation: " << (m_observation)
		<< ", reward: " << m_reward << std::endl;
	return out.str();
}

char LightSensor::read_serial() {
	// Reads a character from the serial line. Blocks until it finds something.
	int read_count = 0;
	char this_value;
	do {
		read_count = read(infd, &this_value, 1);
	} while ((read_count != 1) && (usleep(5) || 1));
	return this_value;
}

char* LightSensor::find_value(char* json) {
	// Given a character pointer containing some JSON, looks for the first
	// occurance of "value": and returns a pointer to whatever follows.
	int nest_count = 0;
	short in_quotes = 0;
	short in_escape = 0;
	short found_value = 0;
	short found_brace = 0;
	int index = 0;
	while ((!found_brace) || (nest_count > 0)) {
		if (in_escape) {
			in_escape = 0;
		}
		else if (in_quotes) {
			switch (json[index]) {
				case '"':
					in_quotes = 0;
					if (index > 7) {
						if (
							(json[index-6] == '"') &&
							(json[index-5] == 'v') &&
							(json[index-4] == 'a') &&
							(json[index-3] == 'l') &&
							(json[index-2] == 'u') &&
							(json[index-1] == 'e')
						) {
							found_value = 1;
						}
					}
					break;
				case '\\':
					in_escape = 1;
					break;
				default:
					break;
			}
		}
		else {
			switch (json[index]) {
				case '{':
					found_brace = 1;
					nest_count++;
					break;
				case '}':
					nest_count--;
					break;
				case '"':
					in_quotes = 1;
				case ':':
					if (found_value == 1) {
						found_value = 2;
					}
					break;
				case '-':
				case '0':
				case '1':
				case '2':
				case '3':
				case '4':
				case '5':
				case '6':
				case '7':
				case '8':
				case '9':
					if (found_value == 2) {
						return json+index;
					}
					break;
				default:
					break;
			}
		}
		index++;
	}
	return (char*) 0;
}

int LightSensor::read_value(char* json) {
	// Given a pointer to a positive integer, written as ASCII characters,
	// returns the number it represents.
	int result = 0;
	short keep_looping = 1;
	int index = 0;
	while (keep_looping) {
		result = result * 10;
		switch(json[index]) {
			case '9':
				result++;
			case '8':
				result++;
			case '7':
				result++;
			case '6':
				result++;
			case '5':
				result++;
			case '4':
				result++;
			case '3':
				result++;
			case '2':
				result++;
			case '1':
				result++;
			case '0':
				break;
			default:
				result = result / 10;
				keep_looping = 0;
		}
		index++;
	}
	return result;
}

char* LightSensor::read_json() {
	// Reads characters from the serial line until a while JSON object has been
	// found.
	short in_quote = 0;		// Are we in a string?
	short in_escape = 0;	// Are we escaped with a backslash?
	int index = 0;			// Where abouts we are currently looking in our buffer
	int nesting = 0;		// How many JSON objects we are inside
	int buffer_length = 50;	// The size of our buffer pointer "json"
	int found_braces = 0;	// Make sure we actually find some JSON
	int real_start = 0;		// Used when we discard garbage prefixes
	char this_value;		// The last character we read
	// Allocate a character pointer of length 2, for "{}"
	char* json = (char*) malloc(sizeof(char)*buffer_length);		// The char buffer for our JSON
	char* new_json;

	// Loop until we explicitly exit or fail to read
	while ((found_braces == 0) || (nesting > 0)) {
		// Read a character
		this_value = read_serial();

		// Skip garbage prefixes
		if ((found_braces == 0) && (this_value != '{')) continue;

		// See if our buffer is big enough for it
		if (index + 1 > buffer_length) {
			buffer_length = buffer_length * 2;		// We need to increase our buffer to hold it
			new_json = (char*) realloc(json, sizeof(char)*buffer_length);
			if (new_json) {
				// Successful allocation, use the new buffer
				json = new_json;
			}
			else {
				// Failed to allocate. Abort
				free(json);
				return (char*)0;
			}
		}

		// Store this character in the buffer
		json[index] = this_value;
		index++;
		
		if (in_escape) {
			// Handle escaping properly, since we want to keep proper track
			// of string boundaries
			continue;
		}
		else if (in_quote) {
			// Handle strings properly, since we want to keep proper track
			// of open and close braces
			switch (this_value) {
				case '"':
					in_quote = 0;
					break;
				case '\\':
					in_escape = 1;
				default:
					break;
			}
		}
		else {
			// Catch-all for everything else. We make no attempt to enforce
			// JSON, we just count the braces.
			switch (this_value) {
				case '{':
					nesting++;
					found_braces = 1;
					break;
				case '}':
					nesting--;
					break;
				case ' ':
					break;
				case '"':
					in_quote = 1;
				case ',':
					break;
				case ':':
					break;
				case '-':
				case '0':
				case '1':
				case '2':
				case '3':
				case '4':
				case '5':
				case '6':
				case '7':
				case '8':
				case '9':
				case '.':
				case 'e':
				case 'E':
					break;
			}
		}
		if ((nesting==0) && (found_braces==1)) {
			// Now allocate enough memory to hold the JSON, plus a null
			// terminator
			new_json = (char*) malloc(sizeof(char)*(buffer_length + 1));
			if (new_json) {
				// Successfully allocated memory, let's fill it
				for (index=0; index < buffer_length; index++) {
					new_json[index] = json[index];
				}
				// We don't need this any more
				free(json);
				// Terminate the pointer to turn it into a proper string
				new_json[buffer_length] = '\0';
				// We're done!
				return new_json;
			}
			else {
				// Error allocating memory. Abort.
				free(json);
				return (char*) 0;
			}
		}
		// We've not found a whole JSON object yet, so read some more input
	}
}
