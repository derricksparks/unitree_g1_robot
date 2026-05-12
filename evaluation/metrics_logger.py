class MetricsLogger:
    def __init__(self, payload_mass_kg=0.0, use_ik_arm_control=False):
        self.total_trials = 1
        self.capture_success_count = 0
        self.transfer_success_count = 0
        self.speeds = []
        self.control_decision_times = []
        self.frame_processing_times = []
        self.assist_forces = []
        self.hand_target_errors = []
        self.final_box_error_to_shelf = None
        self.payload_mass_kg = float(payload_mass_kg)
        self.used_grasp_assist = False
        self.use_ik_arm_control = bool(use_ik_arm_control)

    def record_speed(self, speed):
        if speed > 0.0:
            self.speeds.append(float(speed))

    def record_control_decision_time(self, decision_time):
        self.control_decision_times.append(float(decision_time))

    def record_frame_processing_time(self, processing_time):
        self.frame_processing_times.append(float(processing_time))

    def record_assist_force(self, force_norm):
        force_norm = float(force_norm)
        if force_norm > 0.0:
            self.assist_forces.append(force_norm)

    def record_hand_target_error(self, error):
        if error is not None:
            self.hand_target_errors.append(float(error))

    def mark_capture_success(self):
        self.capture_success_count = 1

    def mark_grasp_assist_used(self):
        self.used_grasp_assist = True

    def mark_transfer_success(self, success):
        self.transfer_success_count = int(bool(success))

    def set_final_box_error(self, error):
        self.final_box_error_to_shelf = float(error)

    @property
    def max_speed(self):
        return max(self.speeds, default=0.0)

    @property
    def average_speed(self):
        if not self.speeds:
            return 0.0
        return sum(self.speeds) / len(self.speeds)

    @property
    def max_control_decision_time(self):
        return max(self.control_decision_times, default=0.0)

    @property
    def average_control_decision_time(self):
        if not self.control_decision_times:
            return 0.0
        return sum(self.control_decision_times) / len(self.control_decision_times)

    @property
    def max_frame_processing_time(self):
        return max(self.frame_processing_times, default=0.0)

    @property
    def average_frame_processing_time(self):
        if not self.frame_processing_times:
            return 0.0
        return sum(self.frame_processing_times) / len(self.frame_processing_times)

    @property
    def max_assist_force(self):
        return max(self.assist_forces, default=0.0)

    @property
    def max_hand_target_error(self):
        return max(self.hand_target_errors, default=0.0)

    @property
    def average_hand_target_error(self):
        if not self.hand_target_errors:
            return 0.0
        return sum(self.hand_target_errors) / len(self.hand_target_errors)

    def as_dict(self):
        return {
            "total_trials": self.total_trials,
            "capture_success_count": self.capture_success_count,
            "transfer_success_count": self.transfer_success_count,
            "max_speed": self.max_speed,
            "average_speed": self.average_speed,
            "max_control_decision_time": self.max_control_decision_time,
            "average_control_decision_time": self.average_control_decision_time,
            "max_frame_processing_time": self.max_frame_processing_time,
            "average_frame_processing_time": self.average_frame_processing_time,
            "final_box_error_to_shelf": self.final_box_error_to_shelf,
            "payload_mass_kg": self.payload_mass_kg,
            "used_grasp_assist": self.used_grasp_assist,
            "max_assist_force": self.max_assist_force,
            "use_ik_arm_control": self.use_ik_arm_control,
            "average_hand_target_error": self.average_hand_target_error,
            "max_hand_target_error": self.max_hand_target_error,
        }

    def print_summary(self):
        print("Final metrics summary")
        for key, value in self.as_dict().items():
            print(f"{key}: {value}")
