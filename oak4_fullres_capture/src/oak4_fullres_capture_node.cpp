#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <depthai/depthai.hpp>
#include <opencv2/imgcodecs.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_srvs/srv/trigger.hpp>

namespace fs = std::filesystem;
using namespace std::chrono_literals;

class Oak4FullresCapture : public rclcpp::Node {
 public:
  Oak4FullresCapture() : Node("oak_fullres_capture") {
    declare_parameter<std::string>("output_path", "");
    declare_parameter<std::string>("roi_output_path", "");
    declare_parameter<std::string>("frame_id", "oak_rgb_camera_optical_frame");
    declare_parameter<bool>("enable_device_capture", false);
    declare_parameter<int>("capture_width", 8000);
    declare_parameter<int>("capture_height", 6000);
    declare_parameter<std::string>("focus_mode", "auto_roi");
    declare_parameter<int>("focus_position", 150);
    declare_parameter<int>("focus_roi_x", 3600);
    declare_parameter<int>("focus_roi_y", 2600);
    declare_parameter<int>("focus_roi_width", 800);
    declare_parameter<int>("focus_roi_height", 800);
    declare_parameter<double>("focus_settle_sec", 1.0);
    declare_parameter<double>("frame_timeout_sec", 12.0);
    declare_parameter<int>("png_compression", 3);
    declare_parameter<int>("jpeg_quality", 98);
    declare_parameter<bool>("publish_image", false);

    image_pub_ = create_publisher<sensor_msgs::msg::Image>("~/image_full", rclcpp::QoS(1).reliable());
    info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>("~/camera_info_full", rclcpp::QoS(1).reliable());
    capture_service_ = create_service<std_srvs::srv::Trigger>(
        "~/capture", std::bind(&Oak4FullresCapture::capture, this, std::placeholders::_1, std::placeholders::_2));
    RCLCPP_INFO(get_logger(), "Ready on %s/capture; camera is opened only while handling a request",
                get_fully_qualified_name());
  }

 private:
  struct CaptureResult {
    cv::Mat image;
    sensor_msgs::msg::CameraInfo camera_info;
    int lens_position = 0;
  };

  void capture(const std_srvs::srv::Trigger::Request::SharedPtr,
               std_srvs::srv::Trigger::Response::SharedPtr response) {
    if (busy_.exchange(true)) {
      response->success = false;
      response->message = "full-resolution capture already active";
      return;
    }
    struct BusyReset {
      std::atomic_bool &flag;
      ~BusyReset() { flag.store(false); }
    } reset{busy_};

    try {
      if (!get_parameter("enable_device_capture").as_bool()) {
        throw std::runtime_error("device capture is disabled; enable only after validating the OAK direct Camera pipeline");
      }
      const auto output_path = get_parameter("output_path").as_string();
      if (output_path.empty()) {
        throw std::runtime_error("output_path parameter is empty");
      }
      const fs::path path(output_path);
      if (path.has_parent_path()) {
        fs::create_directories(path.parent_path());
      }

      auto result = capture_frame();
      write_image(path, result.image);
      const auto roi_output_path = get_parameter("roi_output_path").as_string();
      if (!roi_output_path.empty()) {
        const int x = std::clamp(static_cast<int>(get_parameter("focus_roi_x").as_int()), 0,
                                 result.image.cols - 1);
        const int y = std::clamp(static_cast<int>(get_parameter("focus_roi_y").as_int()), 0,
                                 result.image.rows - 1);
        const int width = std::clamp(static_cast<int>(get_parameter("focus_roi_width").as_int()), 1,
                                     result.image.cols - x);
        const int height = std::clamp(static_cast<int>(get_parameter("focus_roi_height").as_int()), 1,
                                      result.image.rows - y);
        const fs::path roi_path(roi_output_path);
        if (roi_path.has_parent_path()) {
          fs::create_directories(roi_path.parent_path());
        }
        write_image(roi_path, result.image(cv::Rect(x, y, width, height)));
      }
      write_camera_info(path.string() + ".camera_info.yaml", result.camera_info, result.lens_position);

      const auto stamp = now();
      result.camera_info.header.stamp = stamp;
      info_pub_->publish(result.camera_info);
      if (get_parameter("publish_image").as_bool()) {
        image_pub_->publish(to_image_message(result.image, stamp, result.camera_info.header.frame_id));
      }

      std::ostringstream message;
      message << "saved " << path << " " << result.image.cols << "x" << result.image.rows
              << " focus=" << result.lens_position;
      response->success = true;
      response->message = message.str();
      RCLCPP_INFO(get_logger(), "%s", response->message.c_str());
    } catch (const std::exception &error) {
      response->success = false;
      response->message = error.what();
      RCLCPP_ERROR(get_logger(), "Full-resolution capture failed: %s", error.what());
    }
  }

  CaptureResult capture_frame() {
    const auto focus_mode = get_parameter("focus_mode").as_string();
    const int requested_focus = std::clamp(static_cast<int>(get_parameter("focus_position").as_int()), 0, 255);
    const double settle_sec = std::max(0.0, get_parameter("focus_settle_sec").as_double());
    const double timeout_sec = std::max(0.1, get_parameter("frame_timeout_sec").as_double());

    auto device = std::make_shared<dai::Device>();
    dai::Pipeline pipeline(device);
    auto camera = pipeline.create<dai::node::Camera>()->build(dai::CameraBoardSocket::CAM_A);
    if (focus_mode == "manual") {
      camera->initialControl.setManualFocus(static_cast<std::uint8_t>(requested_focus));
    } else if (focus_mode != "auto_roi") {
      throw std::runtime_error("focus_mode must be 'manual' or 'auto_roi'");
    }

    const int capture_width = static_cast<int>(get_parameter("capture_width").as_int());
    const int capture_height = static_cast<int>(get_parameter("capture_height").as_int());
    if (capture_width <= 0 || capture_height <= 0) {
      throw std::runtime_error("capture_width and capture_height must be positive");
    }
    dai::Node::Output *camera_output = nullptr;
    if (capture_width >= 8000 && capture_height >= 6000) {
      camera_output = camera->requestFullResolutionOutput(std::nullopt, 1.0F, true);
    } else {
      camera_output = camera->requestOutput(
          {static_cast<std::uint32_t>(capture_width), static_cast<std::uint32_t>(capture_height)},
          std::nullopt, dai::ImgResizeMode::CROP, 1.0F);
    }
    auto image_queue = camera_output->createOutputQueue(2, false);
    auto control_queue = camera->inputControl.createInputQueue();

    pipeline.start();
    if (focus_mode == "auto_roi") {
      const int max_width = static_cast<int>(camera->getMaxWidth());
      const int max_height = static_cast<int>(camera->getMaxHeight());
      const int x = std::clamp(static_cast<int>(get_parameter("focus_roi_x").as_int()), 0, max_width - 1);
      const int y = std::clamp(static_cast<int>(get_parameter("focus_roi_y").as_int()), 0, max_height - 1);
      const int width = std::clamp(static_cast<int>(get_parameter("focus_roi_width").as_int()), 1, max_width - x);
      const int height = std::clamp(static_cast<int>(get_parameter("focus_roi_height").as_int()), 1, max_height - y);
      auto control = std::make_shared<dai::CameraControl>();
      control->setAutoFocusMode(dai::CameraControl::AutoFocusMode::AUTO);
      control->setAutoFocusLensRange(120, 255);
      control->setAutoFocusRegion(static_cast<std::uint16_t>(x), static_cast<std::uint16_t>(y),
                                  static_cast<std::uint16_t>(width), static_cast<std::uint16_t>(height));
      control->setAutoFocusTrigger();
      control_queue->send(control);
    }

    std::this_thread::sleep_for(std::chrono::duration<double>(settle_sec));
    image_queue->tryGetAll<dai::ImgFrame>();
    bool timed_out = false;
    auto frame = image_queue->get<dai::ImgFrame>(std::chrono::duration<double>(timeout_sec), timed_out);
    if (timed_out || frame == nullptr) {
      pipeline.stop();
      pipeline.wait();
      throw std::runtime_error("timed out waiting for camera frame " + std::to_string(capture_width) +
                               "x" + std::to_string(capture_height));
    }

    cv::Mat image = frame->getCvFrame().clone();
    const int lens_position = static_cast<int>(frame->getLensPosition());
    auto calibration = device->readCalibration();
    auto camera_info = make_camera_info(calibration, image.cols, image.rows);
    pipeline.stop();
    pipeline.wait();
    return CaptureResult{std::move(image), std::move(camera_info), lens_position};
  }

  sensor_msgs::msg::CameraInfo make_camera_info(const dai::CalibrationHandler &calibration, int width, int height) {
    sensor_msgs::msg::CameraInfo info;
    info.header.frame_id = get_parameter("frame_id").as_string();
    info.width = static_cast<std::uint32_t>(width);
    info.height = static_cast<std::uint32_t>(height);
    const auto intrinsics = calibration.getCameraIntrinsics(
        dai::CameraBoardSocket::CAM_A, std::make_tuple(width, height));
    for (std::size_t row = 0; row < 3; ++row) {
      for (std::size_t col = 0; col < 3; ++col) {
        info.k[row * 3 + col] = intrinsics.at(row).at(col);
      }
    }
    info.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    info.p = {info.k[0], info.k[1], info.k[2], 0.0, info.k[3], info.k[4], info.k[5], 0.0,
              info.k[6], info.k[7], info.k[8], 0.0};
    const auto distortion = calibration.getDistortionCoefficients(dai::CameraBoardSocket::CAM_A);
    info.d.assign(distortion.begin(), distortion.end());
    info.distortion_model = info.d.size() > 5 ? "rational_polynomial" : "plumb_bob";
    return info;
  }

  void write_image(const fs::path &path, const cv::Mat &image) {
    std::vector<int> options;
    auto extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    if (extension == ".png") {
      options = {cv::IMWRITE_PNG_COMPRESSION,
                 std::clamp(static_cast<int>(get_parameter("png_compression").as_int()), 0, 9)};
    } else if (extension == ".jpg" || extension == ".jpeg") {
      options = {cv::IMWRITE_JPEG_QUALITY,
                 std::clamp(static_cast<int>(get_parameter("jpeg_quality").as_int()), 1, 100)};
    } else {
      throw std::runtime_error("output_path must end in .png, .jpg, or .jpeg");
    }
    if (!cv::imwrite(path.string(), image, options)) {
      throw std::runtime_error("OpenCV could not write " + path.string());
    }
  }

  static void write_camera_info(const fs::path &path, const sensor_msgs::msg::CameraInfo &info,
                                int lens_position) {
    std::ofstream stream(path);
    if (!stream) {
      throw std::runtime_error("could not write camera info sidecar " + path.string());
    }
    stream << std::setprecision(12);
    stream << "camera_name: " << info.header.frame_id << "\n";
    stream << "image_width: " << info.width << "\n";
    stream << "image_height: " << info.height << "\n";
    stream << "lens_position: " << lens_position << "\n";
    stream << "distortion_model: " << info.distortion_model << "\n";
    stream << "K: [";
    for (std::size_t index = 0; index < info.k.size(); ++index) {
      stream << (index == 0 ? "" : ", ") << info.k[index];
    }
    stream << "]\nD: [";
    for (std::size_t index = 0; index < info.d.size(); ++index) {
      stream << (index == 0 ? "" : ", ") << info.d[index];
    }
    stream << "]\n";
  }

  static sensor_msgs::msg::Image to_image_message(const cv::Mat &image, const rclcpp::Time &stamp,
                                                   const std::string &frame_id) {
    sensor_msgs::msg::Image message;
    message.header.stamp = stamp;
    message.header.frame_id = frame_id;
    message.height = static_cast<std::uint32_t>(image.rows);
    message.width = static_cast<std::uint32_t>(image.cols);
    message.encoding = "bgr8";
    message.is_bigendian = false;
    message.step = static_cast<std::uint32_t>(image.cols * image.elemSize());
    message.data.assign(image.datastart, image.dataend);
    return message;
  }

  std::atomic_bool busy_{false};
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr info_pub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr capture_service_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Oak4FullresCapture>());
  rclcpp::shutdown();
  return 0;
}
