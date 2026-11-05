#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include <mujoco/mujoco.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_srvs/srv/trigger.hpp>

namespace jgarm_sim_cpp {

class MujocoSimNode final : public rclcpp::Node {
 public:
  MujocoSimNode()
      : Node("mujoco_sim_node") {
    model_path_ = declare_parameter<std::string>("model_path", "");
    cycle_us_ = declare_parameter<int>("cycle_us", 1000);
    state_publish_divider_ =
        declare_parameter<int>("state_publish_divider", 2);
    command_timeout_s_ = declare_parameter<double>("command_timeout_s", 0.1);

    if (cycle_us_ <= 0 || state_publish_divider_ <= 0
        || command_timeout_s_ <= 0.0) {
      throw std::invalid_argument(
          "Simulation timing parameters must be positive");
    }

    std::array<char, 1000> error{};
    model_ = mj_loadXML(model_path_.c_str(), nullptr, error.data(), error.size());
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
    if (model_->nq != kJointCount || model_->nv != kJointCount
        || model_->nu != kJointCount) {
      throw std::runtime_error(
          "Expected six MuJoCo joints, velocities, and actuators");
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
    }

    command_subscription_ =
        create_subscription<std_msgs::msg::Float64MultiArray>(
            "/arm/control/effort_cmd", rclcpp::QoS(1),
            [this](const std_msgs::msg::Float64MultiArray::SharedPtr message) {
              command_callback(*message);
            });
    joint_state_publisher_ =
        create_publisher<sensor_msgs::msg::JointState>(
            "/joint_states", rclcpp::SensorDataQoS());
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
        get_logger(), "MuJoCo model loaded: %s", model_path_.c_str());
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
  static constexpr std::array<const char*, kJointCount> kJointNames{
      "joint_1", "joint_2", "joint_3",
      "joint_4", "joint_5", "joint_6"};

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
      command_effort_[index] =
          std::isfinite(message.data[index]) ? message.data[index] : 0.0;
    }
    last_command_time_ = std::chrono::steady_clock::now();
    command_received_ = true;
  }

  void reset_simulation() {
    mj_resetData(model_, data_);
    command_effort_.fill(0.0);
    command_received_ = false;
    mj_forward(model_, data_);
  }

  void simulation_step() {
    const auto steady_now = std::chrono::steady_clock::now();
    const double command_age =
        std::chrono::duration<double>(
            steady_now - last_command_time_).count();
    const bool command_active =
        command_received_ && command_age <= command_timeout_s_;

    for (std::size_t index = 0; index < kJointCount; ++index) {
      data_->ctrl[actuator_ids_[index]] =
          command_active ? command_effort_[index] : 0.0;
    }
    mj_step(model_, data_);

    if (++publish_counter_ < state_publish_divider_) {
      return;
    }
    publish_counter_ = 0;

    sensor_msgs::msg::JointState state;
    state.header.stamp = now();
    state.name.assign(kJointNames.begin(), kJointNames.end());
    state.position.resize(kJointCount);
    state.velocity.resize(kJointCount);
    state.effort.resize(kJointCount);
    for (std::size_t index = 0; index < kJointCount; ++index) {
      state.position[index] = data_->qpos[qpos_adr_[index]];
      state.velocity[index] = data_->qvel[dof_adr_[index]];
      state.effort[index] = data_->actuator_force[actuator_ids_[index]];
    }
    joint_state_publisher_->publish(state);
  }

  mjModel* model_{nullptr};
  mjData* data_{nullptr};
  std::string model_path_;
  int cycle_us_{1000};
  int state_publish_divider_{2};
  double command_timeout_s_{0.1};
  std::array<int, kJointCount> joint_ids_{};
  std::array<int, kJointCount> actuator_ids_{};
  std::array<int, kJointCount> qpos_adr_{};
  std::array<int, kJointCount> dof_adr_{};
  std::array<std::string, kJointCount> joint_names_{
      "joint_1", "joint_2", "joint_3",
      "joint_4", "joint_5", "joint_6"};
  std::array<double, kJointCount> command_effort_{};
  bool command_received_{false};
  std::chrono::steady_clock::time_point last_command_time_{};
  std::size_t publish_counter_{0};
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr
      command_subscription_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr
      joint_state_publisher_;
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
