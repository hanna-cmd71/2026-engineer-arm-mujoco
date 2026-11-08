#include <array>
#include <cmath>

#include <gtest/gtest.h>

#include "jgarm_sim_cpp/hardware_profile.hpp"

namespace {

constexpr std::array<double, 6> kGravityP{
    0.0, 20.284, -15.112, -0.579, 16.368, 0.0};
constexpr std::array<double, 6> kGravityQ{
    0.0, -60.824, 2.680, 0.380, -0.715, 0.0};
constexpr std::array<double, 6> kGravityBias{
    0.0, -22.244, 1.534, 0.0, -13.608, 0.0};
constexpr std::array<double, 6> kFrictionTorque{
    0.0, 11.6, 7.4, 0.0, 0.75, 0.0};
constexpr std::array<double, 6> kFrictionScale{
    0.2, 0.2, 0.2, 0.2, 0.2, 0.2};
constexpr std::array<double, 6> kBreakawayTorque{
    0.0, 11.25, 10.0, 0.0, 0.4, 0.2};

TEST(HardwareProfileTest, GravityMatchesIdentifiedModel) {
  const std::array<double, 6> zero{};
  const auto zero_load = jgarm_sim_cpp::ComputeGravityLoad(
      zero,
      kGravityP,
      kGravityQ,
      kGravityBias,
      0.35,
      true);
  EXPECT_NEAR(zero_load[1], -17.651, 0.002);
  EXPECT_NEAR(zero_load[2], -14.157, 0.002);
  EXPECT_NEAR(zero_load[4], 2.760, 0.002);

  const std::array<double, 6> power_on{
      0.0, -1.3265, 1.35, 0.0, 0.0, 0.0};
  const auto power_on_load = jgarm_sim_cpp::ComputeGravityLoad(
      power_on,
      kGravityP,
      kGravityQ,
      kGravityBias,
      0.35,
      true);
  EXPECT_NEAR(power_on_load[1], 26.065, 0.005);
  EXPECT_NEAR(power_on_load[2], -14.081, 0.005);
  EXPECT_NEAR(power_on_load[4], 2.754, 0.005);
}

TEST(HardwareProfileTest, FrictionMatchesIdentifiedModel) {
  const std::array<double, 6> velocity{
      0.0, 0.2, -0.2, 0.0, 0.0, 0.0};
  const auto load = jgarm_sim_cpp::ComputeFrictionLoad(
      velocity,
      kFrictionTorque,
      kFrictionScale,
      kBreakawayTorque,
      0.02,
      true,
      false);
  EXPECT_NEAR(load[0], 0.0, 1e-12);
  EXPECT_NEAR(load[1], 11.6 * std::tanh(1.0), 1e-12);
  EXPECT_NEAR(load[2], -7.4 * std::tanh(1.0), 1e-12);
  EXPECT_NEAR(load[3], 0.0, 1e-12);
}

TEST(HardwareProfileTest, BreakawayCanBeDisabled) {
  const std::array<double, 6> velocity{
      0.0, 0.2, -0.2, 0.0, 0.0, 0.0};
  const auto loaded = jgarm_sim_cpp::ComputeFrictionLoad(
      velocity,
      kFrictionTorque,
      kFrictionScale,
      kBreakawayTorque,
      0.02,
      true,
      true);
  EXPECT_NEAR(
      loaded[1],
      11.6 * std::tanh(1.0) + 11.25 * std::tanh(10.0),
      1e-12);

  const auto disabled = jgarm_sim_cpp::ComputeFrictionLoad(
      velocity,
      kFrictionTorque,
      kFrictionScale,
      kBreakawayTorque,
      0.02,
      false,
      true);
  EXPECT_NEAR(disabled[1], 0.0, 1e-12);
}

}  // namespace
