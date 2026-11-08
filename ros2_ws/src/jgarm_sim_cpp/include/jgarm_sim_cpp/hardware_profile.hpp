#ifndef JGARM_SIM_CPP_HARDWARE_PROFILE_HPP_
#define JGARM_SIM_CPP_HARDWARE_PROFILE_HPP_

#include <array>
#include <cmath>
#include <cstddef>

namespace jgarm_sim_cpp {

inline std::array<double, 6> ComputeGravityLoad(
    const std::array<double, 6>& position,
    const std::array<double, 6>& p,
    const std::array<double, 6>& q,
    const std::array<double, 6>& bias,
    double j5_config_weight,
    bool j5_conservative_coupling) {
  std::array<double, 6> result = bias;
  const double theta_1 = position[1];
  const double theta_2 = theta_1 + position[2];
  const double theta_3 = theta_2 - position[4];
  const double phi_5 = j5_conservative_coupling
      ? -position[4]
      : j5_config_weight * theta_2 - position[4];

  const auto cosine_sum = [&](double theta, std::size_t index) {
    return p[index] * std::cos(theta) + q[index] * std::sin(theta);
  };
  result[1] += cosine_sum(theta_1, 1) + cosine_sum(theta_2, 2)
               + cosine_sum(theta_3, 3);
  result[2] += cosine_sum(theta_2, 2) + cosine_sum(theta_3, 3);
  result[4] += cosine_sum(phi_5, 4);
  if (j5_conservative_coupling) {
    result[4] += p[3];
    result[4] -= cosine_sum(theta_3, 3);
  }
  return result;
}

inline std::array<double, 6> ComputeFrictionLoad(
    const std::array<double, 6>& velocity,
    const std::array<double, 6>& friction_torque,
    const std::array<double, 6>& friction_velocity_scale,
    const std::array<double, 6>& breakaway_torque,
    double breakaway_velocity_scale,
    bool plant_friction_enabled,
    bool plant_breakaway_enabled) {
  std::array<double, 6> result{};
  if (!plant_friction_enabled) {
    return result;
  }
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index] = friction_torque[index]
                    * std::tanh(
                        velocity[index] / friction_velocity_scale[index]);
    if (plant_breakaway_enabled) {
      result[index] += breakaway_torque[index]
                       * std::tanh(
                           velocity[index] / breakaway_velocity_scale);
    }
  }
  return result;
}

}  // namespace jgarm_sim_cpp

#endif  // JGARM_SIM_CPP_HARDWARE_PROFILE_HPP_
