#pragma once

// Macro-action parameters
#define BELIEF_SAMPLES 100

// Define simulation.
#if defined SIM_LightDark
  #include "core/simulations/LightDark.h"
  typedef simulations::LightDark ExpSimulation;
#elif defined SIM_LightDarkCirc
  #include "core/simulations/LightDarkCirc.h"
  typedef simulations::LightDarkCirc ExpSimulation;
#elif defined SIM_PuckPush
  #include "core/simulations/PuckPush.h"
  typedef simulations::PuckPush ExpSimulation;
#elif defined SIM_VdpTag
  #include "core/simulations/VdpTag.h"
  typedef simulations::VdpTag ExpSimulation;
#elif defined SIM_DriveHard
  #include "core/simulations/DriveHard.h"
  typedef simulations::DriveHard ExpSimulation;
#endif
