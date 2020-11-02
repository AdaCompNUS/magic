// Macro-action parameters
#define BELIEF_SAMPLES 100

// Define simulation.
#ifdef SIM_CornerNav
  #include "core/simulations/CornerNav.h"
  typedef simulations::CornerNav ExpSimulation;
#elif defined SIM_LightDark
  #include "core/simulations/LightDark.h"
  typedef simulations::LightDark ExpSimulation;
#elif defined SIM_IntentionTag
  #include "core/simulations/IntentionTag.h"
  typedef simulations::IntentionTag ExpSimulation;
#elif defined SIM_PuckPush
  #include "core/simulations/PuckPush.h"
  typedef simulations::PuckPush ExpSimulation;
#elif defined SIM_VdpTag
  #include "core/simulations/VdpTag.h"
  typedef simulations::VdpTag ExpSimulation;
#endif

// Define belief.
#include "core/Belief.h"
typedef Belief<ExpSimulation> ExpBelief;
