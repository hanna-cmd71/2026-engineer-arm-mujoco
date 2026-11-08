# PID Controller 
#@author: hanner
#@note: python PID控制器，PD控制输出力矩控制达到指定位置

class PIDController:
    def __init__(self, kp,ki,kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) /dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output