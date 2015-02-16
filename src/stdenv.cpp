#include <cassert>
#include "stdenv.hpp"

// Defaults
const unsigned int StdEnv::cDefaultInBits  = 8;
const unsigned int StdEnv::cDefaultOutBits = 8;

StdEnv::StdEnv(options_t &options) {
  // Set up the input/output bit sizes
  getOption(options, "inbits",  cDefaultInBits,  m_inbits);
  getOption(options, "outbits", cDefaultOutBits, m_outbits);

  // FIXME: Use 1 byte, for simplicity
  assert(m_inbits  <= 8);
  assert(m_outbits <= 8);

  // Open stdio in binary mode
  fin  = fopen("/dev/stdin",  "rb");
  fout = fopen("/dev/stdout", "ab");
  assert(fin  != NULL);
  assert(fout != NULL);

  // Initial percept
  m_observation = 0;
  m_reward      = 0;
}

void StdEnv::performAction(const action_t action) {
  assert(isValidAction(action));
  m_action = action; // Save action

  // Send output
  putc(action, fout);

  // Read input and reward
  m_observation = getc(fin);
  m_reward      = getc(fin);
}

std::string StdEnv::print() const {
  std::ostringstream out;
  out << "prediction: " << m_action
      << ", observation: " << m_observation
      << ", reward: " << m_reward << std::endl;
  return out.str();
}
