#ifndef __STDENV_HPP__
#define __STDENV_HPP__

#include "environment.hpp"

/**
 * Send/receive arbitrary bit strings over stdio
 */
class StdEnv : public Environment {
public:

  // set up the initial environment percept
  StdEnv(options_t &options);

  virtual void performAction(const action_t action);

  virtual action_t  maxAction()      const { return 1; }
  virtual percept_t maxObservation() const { return 1; }
  virtual percept_t maxReward()      const { return 1; }

  virtual std::string print() const;

private:
  /** Number of bits to input/output. Default values are
   * StdEnv::cDefaultInBits and StdEnv::cDefaultOutBits. */
  unsigned int m_inbits;
  unsigned int m_outbits;

  /** Default values for StdEnv::m_inbits and StdEnv::m_outbits. */
  static const unsigned int cDefaultInBits;
  static const unsigned int cDefaultOutBits;

  /** File handles */
  FILE *fin;
  FILE *fout;
};

#endif // __STDENV_HPP__
