#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <depthai/depthai.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
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
    declare_parameter<std::string>("capture_backend", "raw10_host");
    declare_parameter<std::string>("focus_mode", "manual");
    declare_parameter<int>("focus_position", 150);
    declare_parameter<int>("focus_roi_x", 3600);
    declare_parameter<int>("focus_roi_y", 2600);
    declare_parameter<int>("focus_roi_width", 800);
    declare_parameter<int>("focus_roi_height", 800);
    declare_parameter<double>("focus_settle_sec", 1.0);
    declare_parameter<double>("frame_timeout_sec", 12.0);
    declare_parameter<int>("raw_exposure_us", 30000);
    declare_parameter<int>("raw_iso", 800);
    declare_parameter<double>("raw_black_percentile", 0.5);
    declare_parameter<double>("raw_highlight_percentile", 99.7);
    declare_parameter<double>("raw_gamma", 2.2);
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
    std::string backend = "isp";
    int exposure_us = 0;
    int iso = 0;
    int black_level = 0;
    double red_gain = 1.0;
    double blue_gain = 1.0;
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
      write_camera_info(path.string() + ".camera_info.yaml", result);

      const auto stamp = now();
      result.camera_info.header.stamp = stamp;
      info_pub_->publish(result.camera_info);
      if (get_parameter("publish_image").as_bool()) {
        image_pub_->publish(to_image_message(result.image, stamp, result.camera_info.header.frame_id));
      }

      std::ostringstream message;
      message << "saved " << path << " " << result.image.cols << "x" << result.image.rows
              << " backend=" << result.backend << " focus=" << result.lens_position;
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
    const int capture_width = static_cast<int>(get_parameter("capture_width").as_int());
    const int capture_height = static_cast<int>(get_parameter("capture_height").as_int());
    if (capture_width <= 0 || capture_height <= 0) {
      throw std::runtime_error("capture_width and capture_height must be positive");
    }

    const auto backend = get_parameter("capture_backend").as_string();
    if (backend == "raw10_host" && capture_width == 8000 && capture_height == 6000) {
      return capture_raw10(capture_width, capture_height, requested_focus, focus_mode, settle_sec, timeout_sec);
    }
    if (backend != "isp" && backend != "raw10_host") {
      throw std::runtime_error("capture_backend must be 'raw10_host' or 'isp'");
    }
    return capture_isp(capture_width, capture_height, requested_focus, focus_mode, settle_sec, timeout_sec);
  }

  CaptureResult capture_isp(int capture_width, int capture_height, int requested_focus,
                            const std::string &focus_mode, double settle_sec, double timeout_sec) {
    auto device = std::make_shared<dai::Device>();
    dai::Pipeline pipeline(device);
    auto camera = pipeline.create<dai::node::Camera>()->build(dai::CameraBoardSocket::CAM_A);
    if (focus_mode == "manual") {
      camera->initialControl.setManualFocus(static_cast<std::uint8_t>(requested_focus));
    } else if (focus_mode != "auto_roi") {
      throw std::runtime_error("focus_mode must be 'manual' or 'auto_roi'");
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
    CaptureResult result;
    result.image = std::move(image);
    result.camera_info = std::move(camera_info);
    result.lens_position = lens_position;
    return result;
  }

  CaptureResult capture_raw10(int capture_width, int capture_height, int requested_focus,
                              const std::string &focus_mode, double settle_sec, double timeout_sec) {
    if (focus_mode != "manual") {
      throw std::runtime_error(
          "8K raw10_host capture requires focus_mode=manual; autofocus depends on the crashing ISP path");
    }
    const int exposure_us = std::clamp(static_cast<int>(get_parameter("raw_exposure_us").as_int()), 1, 33000);
    const int iso = std::clamp(static_cast<int>(get_parameter("raw_iso").as_int()), 100, 1600);

    auto device = std::make_shared<dai::Device>();
    dai::Pipeline pipeline(device);
    auto camera = pipeline.create<dai::node::Camera>()->build(
        dai::CameraBoardSocket::CAM_A,
        std::make_pair(static_cast<std::uint32_t>(capture_width), static_cast<std::uint32_t>(capture_height)),
        18.0F);
    camera->initialControl.setManualExposure(static_cast<std::uint32_t>(exposure_us),
                                             static_cast<std::uint32_t>(iso));
    camera->initialControl.setManualFocus(static_cast<std::uint8_t>(requested_focus));
    auto raw_queue = camera->raw.createOutputQueue(2, false);

    pipeline.start();
    std::this_thread::sleep_for(std::chrono::duration<double>(settle_sec));
    raw_queue->tryGetAll<dai::ImgFrame>();
    bool timed_out = false;
    auto frame = raw_queue->get<dai::ImgFrame>(std::chrono::duration<double>(timeout_sec), timed_out);
    if (timed_out || frame == nullptr) {
      pipeline.stop();
      pipeline.wait();
      throw std::runtime_error("timed out waiting for 8000x6000 RAW10 frame");
    }
    if (frame->getType() != dai::ImgFrame::Type::RAW10) {
      throw std::runtime_error("camera returned a non-RAW10 frame");
    }

    const int width = static_cast<int>(frame->getWidth());
    const int height = static_cast<int>(frame->getHeight());
    const auto stride = static_cast<std::size_t>(frame->getStride());
    const auto &data = frame->getData();
    const auto packed_row_bytes = static_cast<std::size_t>(width / 4) * 5;
    if (width != capture_width || height != capture_height || width % 4 != 0 ||
        stride < packed_row_bytes || data.size() < stride * static_cast<std::size_t>(height)) {
      throw std::runtime_error("invalid RAW10 frame layout");
    }

    std::array<std::uint64_t, 1024> histogram{};
    std::array<long double, 4> phase_sum{};
    std::array<std::uint64_t, 4> phase_count{};
    for (int y = 0; y < height; ++y) {
      const auto *row = data.data() + static_cast<std::size_t>(y) * stride;
      for (int x = 0; x < width; x += 4) {
        const auto pixels = unpack_raw10(row + static_cast<std::size_t>(x / 4) * 5);
        for (int offset = 0; offset < 4; ++offset) {
          const auto value = pixels[static_cast<std::size_t>(offset)];
          ++histogram[value];
          const auto phase = static_cast<std::size_t>(((y & 1) << 1) | ((x + offset) & 1));
          phase_sum[phase] += value;
          ++phase_count[phase];
        }
      }
    }

    const int black_level = percentile_bin(histogram, get_parameter("raw_black_percentile").as_double());
    const double red_mean = static_cast<double>(phase_sum[0] / phase_count[0]) - black_level;
    const double green_mean = 0.5 * (static_cast<double>(phase_sum[1] / phase_count[1]) +
                                     static_cast<double>(phase_sum[2] / phase_count[2])) - black_level;
    const double blue_mean = static_cast<double>(phase_sum[3] / phase_count[3]) - black_level;
    const double red_gain = std::clamp(green_mean / std::max(1.0, red_mean), 0.25, 8.0);
    const double blue_gain = std::clamp(green_mean / std::max(1.0, blue_mean), 0.25, 8.0);

    cv::Mat bayer16(height, width, CV_16UC1);
    const double sensor_scale = 65535.0 / std::max(1, 1023 - black_level);
    for (int y = 0; y < height; ++y) {
      const auto *row = data.data() + static_cast<std::size_t>(y) * stride;
      auto *output = bayer16.ptr<std::uint16_t>(y);
      for (int x = 0; x < width; x += 4) {
        const auto pixels = unpack_raw10(row + static_cast<std::size_t>(x / 4) * 5);
        for (int offset = 0; offset < 4; ++offset) {
          output[x + offset] = static_cast<std::uint16_t>(std::clamp(
              (static_cast<int>(pixels[static_cast<std::size_t>(offset)]) - black_level) * sensor_scale,
              0.0, 65535.0));
        }
      }
    }

    cv::Mat color16;
    cv::cvtColor(bayer16, color16, cv::COLOR_BayerRG2BGR);
    bayer16.release();
    const std::array<double, 3> gains = {blue_gain, 1.0, red_gain};
    std::array<std::uint64_t, 4096> highlight_histogram{};
    for (int y = 0; y < height; ++y) {
      const auto *row = color16.ptr<cv::Vec<std::uint16_t, 3>>(y);
      for (int x = 0; x < width; ++x) {
        for (int channel = 0; channel < 3; ++channel) {
          const int value = std::clamp(static_cast<int>(row[x][channel] * gains[channel]), 0, 65535);
          ++highlight_histogram[static_cast<std::size_t>(value >> 4)];
        }
      }
    }
    const int highlight = std::max(
        16, percentile_bin(highlight_histogram, get_parameter("raw_highlight_percentile").as_double()) << 4);
    const double gamma = std::max(0.1, get_parameter("raw_gamma").as_double());
    cv::Mat image(height, width, CV_8UC3);
    for (int y = 0; y < height; ++y) {
      const auto *input = color16.ptr<cv::Vec<std::uint16_t, 3>>(y);
      auto *output = image.ptr<cv::Vec3b>(y);
      for (int x = 0; x < width; ++x) {
        for (int channel = 0; channel < 3; ++channel) {
          const double normalized = std::clamp(input[x][channel] * gains[channel] / highlight, 0.0, 1.0);
          output[x][channel] = static_cast<std::uint8_t>(
              std::lround(255.0 * std::pow(normalized, 1.0 / gamma)));
        }
      }
    }
    color16.release();

    auto calibration = device->readCalibration();
    auto camera_info = make_camera_info(calibration, width, height);
    pipeline.stop();
    pipeline.wait();
    CaptureResult result;
    result.image = std::move(image);
    result.camera_info = std::move(camera_info);
    result.lens_position = requested_focus;
    result.backend = "raw10_host";
    result.exposure_us = exposure_us;
    result.iso = iso;
    result.black_level = black_level;
    result.red_gain = red_gain;
    result.blue_gain = blue_gain;
    return result;
  }

  static std::array<std::uint16_t, 4> unpack_raw10(const std::uint8_t *packed) {
    return {static_cast<std::uint16_t>((packed[0] << 2) | (packed[4] & 0x03)),
            static_cast<std::uint16_t>((packed[1] << 2) | ((packed[4] >> 2) & 0x03)),
            static_cast<std::uint16_t>((packed[2] << 2) | ((packed[4] >> 4) & 0x03)),
            static_cast<std::uint16_t>((packed[3] << 2) | ((packed[4] >> 6) & 0x03))};
  }

  template <std::size_t Size>
  static int percentile_bin(const std::array<std::uint64_t, Size> &histogram, double percentile) {
    const auto total = std::accumulate(histogram.begin(), histogram.end(), std::uint64_t{0});
    const auto target = static_cast<std::uint64_t>(
        std::clamp(percentile, 0.0, 100.0) * static_cast<long double>(total) / 100.0L);
    std::uint64_t cumulative = 0;
    for (std::size_t index = 0; index < histogram.size(); ++index) {
      cumulative += histogram[index];
      if (cumulative >= target) {
        return static_cast<int>(index);
      }
    }
    return static_cast<int>(Size - 1);
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

  static void write_camera_info(const fs::path &path, const CaptureResult &result) {
    const auto &info = result.camera_info;
    std::ofstream stream(path);
    if (!stream) {
      throw std::runtime_error("could not write camera info sidecar " + path.string());
    }
    stream << std::setprecision(12);
    stream << "camera_name: " << info.header.frame_id << "\n";
    stream << "image_width: " << info.width << "\n";
    stream << "image_height: " << info.height << "\n";
    stream << "lens_position: " << result.lens_position << "\n";
    stream << "capture_backend: " << result.backend << "\n";
    stream << "exposure_us: " << result.exposure_us << "\n";
    stream << "iso: " << result.iso << "\n";
    stream << "raw_black_level: " << result.black_level << "\n";
    stream << "raw_red_gain: " << result.red_gain << "\n";
    stream << "raw_blue_gain: " << result.blue_gain << "\n";
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
