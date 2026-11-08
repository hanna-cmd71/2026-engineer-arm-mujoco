#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <mujoco/mujoco.h>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/transform_broadcaster.hpp>

#include "jgarm_sim_cpp/hardware_profile.hpp"

namespace jgarm_sim_cpp {

enum class VectorConstraint { kNone, kNonNegative, kPositive };

class MujocoSimNode final : public rclcpp::Node {
 public:
  MujocoSimNode()
      : Node("mujoco_sim_node") {
    model_path_ = declare_parameter<std::string>("model_path", "");
    cycle_us_ = declare_parameter<int>("cycle_us", 1000);
    command_timeout_s_ = declare_parameter<double>("command_timeout_s", 0.1);
    mobile_command_timeout_s_ = declare_parameter<double>(
        "mobile_base.command_timeout_s", 0.25);
    mobile_max_linear_velocity_ = declare_parameter<double>(
        "mobile_base.max_linear_velocity", 0.5);
    mobile_max_angular_velocity_ = declare_parameter<double>(
        "mobile_base.max_angular_velocity", 1.5);
    profile_id_ = declare_parameter<std::string>(
        "hardware_profile.id", "reatrm_engineer");
    const double state_rate_hz = declare_parameter<double>(
        "hardware_profile.state_rate_hz", 111.111111);
    plant_gravity_enabled_ = declare_parameter<bool>(
        "hardware_profile.plant_gravity_enabled", false);
    plant_friction_enabled_ = declare_parameter<bool>(
        "hardware_profile.plant_friction_enabled", true);
    plant_breakaway_enabled_ = declare_parameter<bool>(
        "hardware_profile.plant_breakaway_enabled", true);
    startup_hold_enabled_ = declare_parameter<bool>(
        "startup_hold_enabled", true);

    effort_limits_ = declare_vector6(
        "hardware_profile.effort_limits_nm",
        {10.0, 63.0, 35.0, 10.0, 6.0, 3.0},
        VectorConstraint::kPositive);
    gravity_p_ = declare_vector6(
        "hardware_profile.gravity_p_nm",
        {0.0, 20.284, -15.112, -0.579, 16.368, 0.0});
    gravity_q_ = declare_vector6(
        "hardware_profile.gravity_q_nm",
        {0.0, -60.824, 2.680, 0.380, -0.715, 0.0});
    gravity_bias_ = declare_vector6(
        "hardware_profile.gravity_bias_nm",
        {0.0, -22.244, 1.534, 0.0, -13.608, 0.0});
    friction_torque_ = declare_vector6(
        "hardware_profile.friction_torque_nm",
        {0.0, 11.6, 7.4, 0.0, 0.75, 0.0},
        VectorConstraint::kNonNegative);
    joint_viscous_damping_ = declare_vector6(
        "hardware_profile.joint_viscous_damping_nm_s_rad",
        {0.05, 0.05, 0.05, 0.05, 0.05, 0.05},
        VectorConstraint::kNonNegative);
    joint_armature_ = declare_vector6(
        "hardware_profile.joint_armature_kg_m2",
        {0.001, 0.001, 0.001, 0.001, 0.001, 0.001},
        VectorConstraint::kNonNegative);
    actuator_time_constant_ = declare_vector6(
        "hardware_profile.actuator_time_constant_s",
        {0.005, 0.005, 0.005, 0.005, 0.005, 0.005},
        VectorConstraint::kPositive);
    friction_velocity_scale_ = declare_vector6(
        "hardware_profile.friction_velocity_scale_rad_s",
        {0.2, 0.2, 0.2, 0.2, 0.2, 0.2},
        VectorConstraint::kPositive);
    breakaway_torque_ = declare_vector6(
        "hardware_profile.breakaway_torque_nm",
        {0.0, 11.25, 10.0, 0.0, 0.4, 0.2},
        VectorConstraint::kNonNegative);
    gravity_j5_config_weight_ = declare_parameter<double>(
        "hardware_profile.gravity_j5_config_weight", 0.35);
    gravity_j5_conservative_coupling_ = declare_parameter<bool>(
        "hardware_profile.gravity_j5_conservative_coupling", true);
    breakaway_velocity_scale_ = declare_parameter<double>(
        "hardware_profile.plant_breakaway_velocity_scale_rad_s", 0.02);

    if (cycle_us_ <= 0 || state_publish_divider_ <= 0
        || !std::isfinite(command_timeout_s_) || command_timeout_s_ <= 0.0
        || !std::isfinite(state_rate_hz) || state_rate_hz <= 0.0
        || !std::isfinite(breakaway_velocity_scale_)
        || breakaway_velocity_scale_ <= 0.0
        || !std::isfinite(gravity_j5_config_weight_)
        || gravity_j5_config_weight_ <= 0.0
        || !std::isfinite(mobile_command_timeout_s_)
        || mobile_command_timeout_s_ <= 0.0
        || !std::isfinite(mobile_max_linear_velocity_)
        || mobile_max_linear_velocity_ <= 0.0
        || !std::isfinite(mobile_max_angular_velocity_)
        || mobile_max_angular_velocity_ <= 0.0) {
      throw std::invalid_argument(
          "Simulation timing parameters must be positive");
    }
    state_publish_divider_ = std::max(
        1, static_cast<int>(std::lround(
               1.0e6 / (static_cast<double>(cycle_us_) * state_rate_hz))));

    std::array<char, 1000> error{};
    model_ = mj_loadXML(
        model_path_.c_str(), nullptr, error.data(), error.size());
    if (model_ == nullptr) {
      throw std::runtime_error(
          "Failed to load MuJoCo model: " + std::string(error.data()));
    }
    data_ = mj_makeData(model_);
    if (data_ == nullptr) {
      mj_deleteModel(model_);
      model_ = nullptr;
      throw std::runtime_error("Failed to allocate MuJoCo data");
    }
    if (plant_gravity_enabled_) {
      model_->opt.gravity[0] = 0.0;
      model_->opt.gravity[1] = 0.0;
      model_->opt.gravity[2] = 0.0;
    }
    if (model_->nq < kJointCount || model_->nv < kJointCount
        || model_->nu != kJointCount) {
      throw std::runtime_error(
          "Expected at least six joint coordinates and six actuators");
    }
    for (std::size_t index = 0; index < kEnergyUnitCount; ++index) {
      scene_object_body_ids_[index] = mj_name2id(
          model_, mjOBJ_BODY, kEnergyUnitBodyNames[index]);
      if (scene_object_body_ids_[index] < 0
          || model_->body_mass[scene_object_body_ids_[index]] <= 0.0) {
        throw std::runtime_error(
            "MuJoCo model is missing " + std::string(
                kEnergyUnitBodyNames[index]));
      }
      scene_object_masses_kg_[index] =
          model_->body_mass[scene_object_body_ids_[index]];
    }
    power_on_key_id_ =
        mj_name2id(model_, mjOBJ_KEY, "power_on");
    if (power_on_key_id_ < 0) {
      throw std::runtime_error(
          "MuJoCo model is missing the power_on keyframe");
    }
    mj_resetDataKeyframe(model_, data_, power_on_key_id_);
    mj_forward(model_, data_);

    base_link_body_id_ =
        mj_name2id(model_, mjOBJ_BODY, "base_link");
    if (base_link_body_id_ < 0) {
      throw std::runtime_error("MuJoCo model is missing base_link");
    }
    for (std::size_t index = 0; index < kMobileJointCount; ++index) {
      mobile_joint_ids_[index] = mj_name2id(
          model_, mjOBJ_JOINT, kMobileJointNames[index]);
      if (mobile_joint_ids_[index] < 0) {
        throw std::runtime_error(
            "MuJoCo model is missing " + std::string(
                kMobileJointNames[index]));
      }
      mobile_qpos_adr_[index] =
          model_->jnt_qposadr[mobile_joint_ids_[index]];
      mobile_dof_adr_[index] =
          model_->jnt_dofadr[mobile_joint_ids_[index]];
    }

    for (std::size_t index = 0; index < kJointCount; ++index) {
      joint_ids_[index] =
          mj_name2id(model_, mjOBJ_JOINT, joint_names_[index].c_str());
      actuator_ids_[index] = mj_name2id(
          model_, mjOBJ_ACTUATOR,
          (joint_names_[index] + "_motor").c_str());
      if (joint_ids_[index] < 0 || actuator_ids_[index] < 0) {
        throw std::runtime_error(
            "MuJoCo model is missing " + joint_names_[index]);
      }
      qpos_adr_[index] = model_->jnt_qposadr[joint_ids_[index]];
      dof_adr_[index] = model_->jnt_dofadr[joint_ids_[index]];
      model_->dof_damping[dof_adr_[index]] =
          joint_viscous_damping_[index];
      model_->dof_armature[dof_adr_[index]] = joint_armature_[index];
    }

    command_subscription_ =
        create_subscription<std_msgs::msg::Float64MultiArray>(
            "/arm/control/effort_cmd", rclcpp::QoS(1),
            [this](const std_msgs::msg::Float64MultiArray::SharedPtr message) {
              command_callback(*message);
            });
    mobile_command_subscription_ =
        create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", rclcpp::QoS(1).reliable(),
            [this](const geometry_msgs::msg::Twist::SharedPtr message) {
              mobile_command_callback(*message);
            });
    joint_state_publisher_ =
        create_publisher<sensor_msgs::msg::JointState>(
            "/joint_states", rclcpp::QoS(10).reliable());
    fast_joint_state_publisher_ =
        create_publisher<sensor_msgs::msg::JointState>(
            "/arm/control/joint_state", rclcpp::QoS(1).reliable());
    bias_effort_publisher_ =
        create_publisher<std_msgs::msg::Float64MultiArray>(
            "/arm/debug/mujoco_bias_effort", rclcpp::QoS(10).reliable());
    friction_effort_publisher_ =
        create_publisher<std_msgs::msg::Float64MultiArray>(
            "/arm/debug/mujoco_friction_effort",
            rclcpp::QoS(10).reliable());
    odometry_publisher_ =
        create_publisher<nav_msgs::msg::Odometry>(
            "/odom", rclcpp::QoS(10).reliable());
    transform_broadcaster_ =
        std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    reset_service_ = create_service<std_srvs::srv::Trigger>(
        "/arm/sim/reset",
        [this](
            const std::shared_ptr<std_srvs::srv::Trigger::Request>,
            std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
          reset_simulation();
          response->success = true;
          response->message = "JGarm simulation reset";
        });

    timer_ = create_wall_timer(
        std::chrono::microseconds(cycle_us_),
        std::bind(&MujocoSimNode::simulation_step, this));
    RCLCPP_INFO(
        get_logger(), "MuJoCo model loaded: %s (hardware profile: %s)",
        model_path_.c_str(), profile_id_.c_str());
  }

  ~MujocoSimNode() override {
    if (data_ != nullptr) {
      mj_deleteData(data_);
    }
    if (model_ != nullptr) {
      mj_deleteModel(model_);
    }
  }

 private:
  static constexpr std::size_t kJointCount = 6;
  static constexpr std::size_t kEnergyUnitCount = 6;
  static constexpr std::size_t kMobileJointCount = 7;
  static constexpr std::size_t kWheelCount = 4;
  static constexpr std::size_t kPublishedJointCount =
      kJointCount + kWheelCount;
  static constexpr std::array<const char*, kJointCount> kJointNames{
      "joint_1", "joint_2", "joint_3",
      "joint_4", "joint_5", "joint_6"};
  static constexpr std::array<const char*, kEnergyUnitCount>
      kEnergyUnitBodyNames{
          "energy_unit_01", "energy_unit_02", "energy_unit_03",
          "energy_unit_04", "energy_unit_05", "energy_unit_06"};
  static constexpr std::array<const char*, kMobileJointCount>
      kMobileJointNames{
          "base_x", "base_y", "base_yaw",
          "wheel_front_left_spin", "wheel_front_right_spin",
          "wheel_rear_left_spin", "wheel_rear_right_spin"};
  static constexpr std::array<const char*, kPublishedJointCount>
      kPublishedJointNames{
          "joint_1", "joint_2", "joint_3",
          "joint_4", "joint_5", "joint_6",
          "wheel_front_left_spin", "wheel_front_right_spin",
          "wheel_rear_left_spin", "wheel_rear_right_spin"};

  std::array<double, 6> declare_vector6(
      const std::string& name,
      std::initializer_list<double> defaults,
      VectorConstraint constraint = VectorConstraint::kNone) {
    const std::vector<double> values = declare_parameter<std::vector<double>>(
        name, std::vector<double>(defaults));
    if (values.size() != kJointCount) {
      throw std::invalid_argument(name + " must contain six values");
    }
    std::array<double, 6> result{};
    for (std::size_t index = 0; index < kJointCount; ++index) {
      if (!std::isfinite(values[index])) {
        throw std::invalid_argument(name + " must contain finite values");
      }
      if (constraint == VectorConstraint::kPositive
          && values[index] <= 0.0) {
        throw std::invalid_argument(name + " must contain positive values");
      }
      if (constraint == VectorConstraint::kNonNegative
          && values[index] < 0.0) {
        throw std::invalid_argument(
            name + " must contain non-negative values");
      }
      result[index] = values[index];
    }
    return result;
  }

  std::array<double, 6> compute_gravity_load() const {
    std::array<double, 6> position{};
    for (std::size_t index = 0; index < kJointCount; ++index) {
      position[index] = data_->qpos[qpos_adr_[index]];
    }
    return ComputeGravityLoad(
        position,
        gravity_p_,
        gravity_q_,
        gravity_bias_,
        gravity_j5_config_weight_,
        gravity_j5_conservative_coupling_);
  }

  std::array<double, 6> compute_friction_load() const {
    std::array<double, 6> velocity{};
    for (std::size_t index = 0; index < kJointCount; ++index) {
      velocity[index] = data_->qvel[dof_adr_[index]];
    }
    return ComputeFrictionLoad(
        velocity,
        friction_torque_,
        friction_velocity_scale_,
        breakaway_torque_,
        breakaway_velocity_scale_,
        plant_friction_enabled_,
        plant_breakaway_enabled_);
  }

  void command_callback(
      const std_msgs::msg::Float64MultiArray& message) {
    if (message.data.size() != kJointCount) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "Expected six effort commands, received %zu",
          message.data.size());
      return;
    }
    for (std::size_t index = 0; index < kJointCount; ++index) {
      const double requested =
          std::isfinite(message.data[index]) ? message.data[index] : 0.0;
      command_effort_[index] =
          std::clamp(requested, -effort_limits_[index], effort_limits_[index]);
    }
    last_command_time_ = std::chrono::steady_clock::now();
    command_received_ = true;
  }

  void mobile_command_callback(
      const geometry_msgs::msg::Twist& message) {
    const double vx = std::isfinite(message.linear.x)
        ? std::clamp(
            message.linear.x,
            -mobile_max_linear_velocity_,
            mobile_max_linear_velocity_)
        : 0.0;
    const double vy = std::isfinite(message.linear.y)
        ? std::clamp(
            message.linear.y,
            -mobile_max_linear_velocity_,
            mobile_max_linear_velocity_)
        : 0.0;
    const double wz = std::isfinite(message.angular.z)
        ? std::clamp(
            message.angular.z,
            -mobile_max_angular_velocity_,
            mobile_max_angular_velocity_)
        : 0.0;
    mobile_command_[0] = vx;
    mobile_command_[1] = vy;
    mobile_command_[2] = wz;
    last_mobile_command_time_ = std::chrono::steady_clock::now();
    mobile_command_received_ = true;
  }

  void update_mobile_base() {
    const auto steady_now = std::chrono::steady_clock::now();
    const double command_age =
        std::chrono::duration<double>(
            steady_now - last_mobile_command_time_).count();
    const bool command_active =
        mobile_command_received_
        && command_age <= mobile_command_timeout_s_;
    const std::array<double, 3> target{
        command_active ? mobile_command_[0] : 0.0,
        command_active ? mobile_command_[1] : 0.0,
        command_active ? mobile_command_[2] : 0.0};
    const double dt = static_cast<double>(cycle_us_) * 1.0e-6;
    const double alpha = 1.0 - std::exp(-dt / 0.08);
    for (std::size_t index = 0; index < 3; ++index) {
      mobile_velocity_[index] +=
          alpha * (target[index] - mobile_velocity_[index]);
    }

    const double yaw = data_->qpos[mobile_qpos_adr_[2]];
    const double cosine = std::cos(yaw);
    const double sine = std::sin(yaw);
    data_->qvel[mobile_dof_adr_[0]] =
        cosine * mobile_velocity_[0] - sine * mobile_velocity_[1];
    data_->qvel[mobile_dof_adr_[1]] =
        sine * mobile_velocity_[0] + cosine * mobile_velocity_[1];
    data_->qvel[mobile_dof_adr_[2]] = mobile_velocity_[2];

    constexpr double kWheelRadius = 0.045;
    constexpr double kHalfLength = 0.18;
    constexpr double kHalfWidth = 0.15;
    const double rotation_term =
        (kHalfLength + kHalfWidth) * mobile_velocity_[2];
    const std::array<double, 4> wheel_linear_velocity{
        mobile_velocity_[0] - mobile_velocity_[1] - rotation_term,
        mobile_velocity_[0] + mobile_velocity_[1] + rotation_term,
        mobile_velocity_[0] + mobile_velocity_[1] - rotation_term,
        mobile_velocity_[0] - mobile_velocity_[1] + rotation_term};
    for (std::size_t index = 0; index < 4; ++index) {
      data_->qvel[mobile_dof_adr_[3 + index]] =
          wheel_linear_velocity[index] / kWheelRadius;
    }
  }

  void publish_mobile_state() {
    const rclcpp::Time stamp = now();
    geometry_msgs::msg::TransformStamped transform;
    transform.header.stamp = stamp;
    transform.header.frame_id = "world";
    transform.child_frame_id = "base_link";
    transform.transform.translation.x =
        data_->xpos[base_link_body_id_ * 3];
    transform.transform.translation.y =
        data_->xpos[base_link_body_id_ * 3 + 1];
    transform.transform.translation.z =
        data_->xpos[base_link_body_id_ * 3 + 2];
    transform.transform.rotation.w =
        data_->xquat[base_link_body_id_ * 4];
    transform.transform.rotation.x =
        data_->xquat[base_link_body_id_ * 4 + 1];
    transform.transform.rotation.y =
        data_->xquat[base_link_body_id_ * 4 + 2];
    transform.transform.rotation.z =
        data_->xquat[base_link_body_id_ * 4 + 3];
    transform_broadcaster_->sendTransform(transform);

    nav_msgs::msg::Odometry odometry;
    odometry.header.stamp = stamp;
    odometry.header.frame_id = "world";
    odometry.child_frame_id = "base_link";
    odometry.pose.pose.position.x = transform.transform.translation.x;
    odometry.pose.pose.position.y = transform.transform.translation.y;
    odometry.pose.pose.position.z = transform.transform.translation.z;
    odometry.pose.pose.orientation = transform.transform.rotation;
    odometry.twist.twist.linear.x = mobile_velocity_[0];
    odometry.twist.twist.linear.y = mobile_velocity_[1];
    odometry.twist.twist.angular.z = mobile_velocity_[2];
    odometry_publisher_->publish(odometry);
  }

  void reset_simulation() {
    mj_resetDataKeyframe(model_, data_, power_on_key_id_);
    command_effort_.fill(0.0);
    effective_effort_.fill(0.0);
    command_received_ = false;
    mobile_command_.fill(0.0);
    mobile_velocity_.fill(0.0);
    mobile_command_received_ = false;
    mj_forward(model_, data_);
  }

  void simulation_step() {
    const auto steady_now = std::chrono::steady_clock::now();
    const double command_age =
        std::chrono::duration<double>(
            steady_now - last_command_time_).count();
    const bool command_active =
        command_received_ && command_age <= command_timeout_s_;

    const std::array<double, 6> gravity_load = compute_gravity_load();
    const std::array<double, 6> friction_load = compute_friction_load();
    for (std::size_t index = 0; index < kJointCount; ++index) {
      const double target_effort =
          command_active
              ? command_effort_[index]
              : (startup_hold_enabled_ && !command_received_
                     ? data_->qfrc_bias[dof_adr_[index]]
                     : 0.0);
      const double alpha =
          static_cast<double>(cycle_us_) * 1.0e-6
          / (actuator_time_constant_[index]
             + static_cast<double>(cycle_us_) * 1.0e-6);
      effective_effort_[index] +=
          alpha * (target_effort - effective_effort_[index]);
      data_->ctrl[actuator_ids_[index]] = effective_effort_[index];
      const double gravity_force =
          plant_gravity_enabled_ ? gravity_load[index] : 0.0;
      data_->qfrc_applied[dof_adr_[index]] =
          -gravity_force - friction_load[index];
    }
    update_mobile_base();
    for (std::size_t index = 0; index < kEnergyUnitCount; ++index) {
      const std::size_t object_force_address =
          static_cast<std::size_t>(scene_object_body_ids_[index]) * 6;
      data_->xfrc_applied[object_force_address + 2] =
          plant_gravity_enabled_ ? -scene_object_masses_kg_[index] * 9.81
                                 : 0.0;
    }
    mj_step(model_, data_);

    sensor_msgs::msg::JointState state;
    state.header.stamp = now();
    state.name.assign(
        kPublishedJointNames.begin(), kPublishedJointNames.end());
    state.position.resize(kPublishedJointCount);
    state.velocity.resize(kPublishedJointCount);
    state.effort.resize(kPublishedJointCount);
    for (std::size_t index = 0; index < kJointCount; ++index) {
      state.position[index] = data_->qpos[qpos_adr_[index]];
      state.velocity[index] = data_->qvel[dof_adr_[index]];
      state.effort[index] = data_->actuator_force[actuator_ids_[index]];
    }
    for (std::size_t index = 0; index < kWheelCount; ++index) {
      const std::size_t state_index = kJointCount + index;
      state.position[state_index] =
          data_->qpos[mobile_qpos_adr_[3 + index]];
      state.velocity[state_index] =
          data_->qvel[mobile_dof_adr_[3 + index]];
      state.effort[state_index] = 0.0;
    }
    if (++fast_publish_counter_ >= 2) {
      fast_publish_counter_ = 0;
      fast_joint_state_publisher_->publish(state);
    }

    if (++publish_counter_ < state_publish_divider_) {
      return;
    }
    publish_counter_ = 0;

    std_msgs::msg::Float64MultiArray bias_effort;
    bias_effort.data.resize(kJointCount);
    std_msgs::msg::Float64MultiArray friction_effort;
    friction_effort.data.resize(kJointCount);
    for (std::size_t index = 0; index < kJointCount; ++index) {
      bias_effort.data[index] =
          plant_gravity_enabled_ ? gravity_load[index]
                                 : data_->qfrc_bias[dof_adr_[index]];
      friction_effort.data[index] = friction_load[index];
    }
    joint_state_publisher_->publish(state);
    bias_effort_publisher_->publish(bias_effort);
    friction_effort_publisher_->publish(friction_effort);
    publish_mobile_state();
  }

  mjModel* model_{nullptr};
  mjData* data_{nullptr};
  std::string model_path_;
  std::string profile_id_;
  int cycle_us_{1000};
  int state_publish_divider_{9};
  double command_timeout_s_{0.1};
  double mobile_command_timeout_s_{0.25};
  double mobile_max_linear_velocity_{0.5};
  double mobile_max_angular_velocity_{1.5};
  bool plant_gravity_enabled_{true};
  bool plant_friction_enabled_{true};
  bool plant_breakaway_enabled_{true};
  bool startup_hold_enabled_{true};
  double gravity_j5_config_weight_{0.35};
  bool gravity_j5_conservative_coupling_{true};
  double breakaway_velocity_scale_{0.02};
  int power_on_key_id_{-1};
  int base_link_body_id_{-1};
  std::array<int, kMobileJointCount> mobile_joint_ids_{};
  std::array<int, kMobileJointCount> mobile_qpos_adr_{};
  std::array<int, kMobileJointCount> mobile_dof_adr_{};
  std::array<double, 3> mobile_command_{};
  std::array<double, 3> mobile_velocity_{};
  std::array<int, kEnergyUnitCount> scene_object_body_ids_{};
  std::array<double, kEnergyUnitCount> scene_object_masses_kg_{};
  std::array<int, kJointCount> joint_ids_{};
  std::array<int, kJointCount> actuator_ids_{};
  std::array<int, kJointCount> qpos_adr_{};
  std::array<int, kJointCount> dof_adr_{};
  std::array<std::string, kJointCount> joint_names_{
      "joint_1", "joint_2", "joint_3",
      "joint_4", "joint_5", "joint_6"};
  std::array<double, kJointCount> command_effort_{};
  std::array<double, kJointCount> effective_effort_{};
  std::array<double, kJointCount> effort_limits_{};
  std::array<double, kJointCount> gravity_p_{};
  std::array<double, kJointCount> gravity_q_{};
  std::array<double, kJointCount> gravity_bias_{};
  std::array<double, kJointCount> friction_torque_{};
  std::array<double, kJointCount> friction_velocity_scale_{};
  std::array<double, kJointCount> joint_viscous_damping_{};
  std::array<double, kJointCount> joint_armature_{};
  std::array<double, kJointCount> actuator_time_constant_{};
  std::array<double, kJointCount> breakaway_torque_{};
  bool command_received_{false};
  bool mobile_command_received_{false};
  std::chrono::steady_clock::time_point last_command_time_{};
  std::chrono::steady_clock::time_point last_mobile_command_time_{};
  std::size_t fast_publish_counter_{0};
  std::size_t publish_counter_{0};
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr
      command_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr
      mobile_command_subscription_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr
      joint_state_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr
      fast_joint_state_publisher_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr
      bias_effort_publisher_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr
      friction_effort_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr
      odometry_publisher_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> transform_broadcaster_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_service_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace jgarm_sim_cpp

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<jgarm_sim_cpp::MujocoSimNode>());
  rclcpp::shutdown();
  return 0;
}
