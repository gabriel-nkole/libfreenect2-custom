/*
 * This file is part of the OpenKinect Project. http://www.openkinect.org
 *
 * Copyright (c) 2011 individual OpenKinect contributors. See the CONTRIB file
 * for details.
 *
 * This code is licensed to you under the terms of the Apache License, version
 * 2.0, or, at your option, the terms of the GNU General Public License,
 * version 2.0. See the APACHE20 and GPL2 files for the text of the licenses,
 * or the following URLs:
 * http://www.apache.org/licenses/LICENSE-2.0
 * http://www.gnu.org/licenses/gpl-2.0.txt
 *
 * If you redistribute this file in source form, modified or unmodified, you
 * may:
 *   1) Leave this header intact and distribute it under the same terms,
 *      accompanying it with the APACHE20 and GPL20 files, or
 *   2) Delete the Apache 2.0 clause and accompany it with the GPL2 file, or
 *   3) Delete the GPL v2 clause and accompany it with the APACHE20 file
 * In all cases you must keep the copyright notice intact and include a copy
 * of the CONTRIB file.
 *
 * Binary distributions must follow the binary distribution requirements of
 * either License.
 */

/** @file Protonect.cpp Main application file. */

#include <iostream>
#include <cstdlib>
#include <signal.h>

/// [headers]
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/logger.h>
/// [headers]
#include <libfreenect2/config.h>
#include "shader_m.h"

#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"

#include <omp.h>
#include <random>
#include <algorithm>

bool protonect_shutdown = false; ///< Whether the running application should shut down.

void sigint_handler(int s) {
  protonect_shutdown = true;
}

bool protonect_paused = false;
libfreenect2::Freenect2Device *devtopause;

//Doing non-trivial things in signal handler is bad. If you want to pause,
//do it in another thread.
//Though libusb operations are generally thread safe, I cannot guarantee
//everything above is thread safe when calling start()/stop() while
//waitForNewFrame().
void sigusr1_handler(int s) {
  if (devtopause == 0)
    return;
/// [pause]
  if (protonect_paused)
    devtopause->start();
  else
    devtopause->stop();
  protonect_paused = !protonect_paused;
/// [pause]
}

//The following demostrates how to create a custom logger
/// [logger]
#include <fstream>
#include <cstdlib>
class MyFileLogger: public libfreenect2::Logger {
private:
  std::ofstream logfile_;
public:
  MyFileLogger(const char *filename) {
    if (filename)
      logfile_.open(filename);
    level_ = Debug;
  }
  bool good() {
    return logfile_.is_open() && logfile_.good();
  }
  virtual void log(Level level, const std::string &message) {
    logfile_ << "[" << libfreenect2::Logger::level2str(level) << "] " << message << std::endl;
  }
};
/// [logger]

/// [main]
/**
 * Main application entry point.
 *
 * Accepted argumemnts:
 * - cpu Perform depth processing with the CPU.
 * - gl  Perform depth processing with OpenGL.
 * - cl  Perform depth processing with OpenCL.
 */



void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

// Camera Controls
float ZPos = 0.0f;
float MovementSpeed = 1.0f;

//glm::vec3 cameraRotation(25.0f, 315.0f, 0.0f);
glm::vec3 cameraRotation(0.0f, 0.0f, 0.0f);
float RotationSpeed = 0.5f;
float LastX = 0.0f;
float LastY = 0.0f;
float DeltaX = 0.0f;
float DeltaY = 0.0f;
bool FirstMouse = true;

float YOffset = 0.0f;
//float ZoomDistance = 2.0f;
float ZoomDistance = 0.1f;
float ZoomSpeed = 0.2f;

// Camera
glm::vec3 Up = glm::vec3(0.0f, 1.0f, 0.0f);
glm::vec3 Position = glm::vec3(0.0f, 0.0f, ZoomDistance);
glm::vec3 Front = glm::vec3(0.0f, 0.0f, -1.0f);
glm::mat4 T(1.0f);
glm::mat4 R(1.0f);


// Constants
const float PI = 3.1415926535897932385f;
const float Deg2Rad = PI/180.0f;

static const int M = 424;
static const int N = 512;
static const int NumVertices = M*N;
const float fx=367.286994337726f;        // Focal length in X and Y
const float fy=367.286855347968f;
const float cx=255.165695200749f;        // Principle point in X and Y
const float cy=211.824600345805f;

const unsigned int histoSize = 360;


int main(int argc, char *argv[]) {
  // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(1280, 720, "LearnOpenGL", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // tell GLFW to capture our mouse
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
  }

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO(); (void)io;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

  // Setup Dear ImGui style
  ImGui::StyleColorsDark();
  //ImGui::StyleColorsLight();

  // Setup Platform/Renderer backends
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 330 core");
  

  glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
  glEnable(GL_DEPTH_TEST);
  //glPolygonOffset(1, 1);
  //glEnable(GL_BLEND);
  //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  Shader meshShader = Shader("examples/shaders/mesh.vert", "examples/shaders/mesh.frag");


  std::string program_path(argv[0]);
  std::cerr << "Version: " << LIBFREENECT2_VERSION << std::endl;
  std::cerr << "Environment variables: LOGFILE=<protonect.log>" << std::endl;
  std::cerr << "Usage: " << program_path << " [-gpu=<id>] [gl | cl | clkde | cuda | cudakde | cpu]" << std::endl;
  std::cerr << "        [-norgb | -nodepth] [-help] [-version]" << std::endl;
  std::cerr << "To pause and unpause: pkill -USR1 Protonect" << std::endl;
  size_t executable_name_idx = program_path.rfind("Protonect");

  std::string binpath = "/";

  if(executable_name_idx != std::string::npos) {
    binpath = program_path.substr(0, executable_name_idx);
  }

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
  // avoid flooing the very slow Windows console with debug messages
  libfreenect2::setGlobalLogger(libfreenect2::createConsoleLogger(libfreenect2::Logger::Info));
#else
  // create a console logger with debug level (default is console logger with info level)
/// [logging]
  libfreenect2::setGlobalLogger(libfreenect2::createConsoleLogger(libfreenect2::Logger::Debug));
/// [logging]
#endif
/// [file logging]
  MyFileLogger *filelogger = new MyFileLogger(getenv("LOGFILE"));
  if (filelogger->good())
    libfreenect2::setGlobalLogger(filelogger);
  else
    delete filelogger;
/// [file logging]

/// [context]
  libfreenect2::Freenect2 freenect2;
  libfreenect2::PacketPipeline *pipeline = 0;

  struct kinect_device {
    std::string serial = "";
    libfreenect2::Freenect2Device *dev = 0;
    libfreenect2::SyncMultiFrameListener *listener;
    libfreenect2::FrameMap frames;
    libfreenect2::Registration* registration;
    libfreenect2::Frame undistorted = libfreenect2::Frame(N, M, 4);
    libfreenect2::Frame registered = libfreenect2::Frame(N, M, 4);
    bool firstFrame = true;
    bool newFrame = false;

    cv::Mat rgb = cv::Mat(M, N, CV_8UC4);
    cv::Mat rgbHD = cv::Mat(1080, 1920, CV_8UC4);
    cv::Mat hsv = cv::Mat(M, N, CV_8UC3);
    cv::Mat depth = cv::Mat(M, N, CV_32FC1);
    cv::Mat depthSmooth = cv::Mat(M, N, CV_32FC1);

    std::vector<float> meshVertices = std::vector<float>(NumVertices*10);
    std::vector<float> meshVertices2 = std::vector<float>(NumVertices*10);
    unsigned int meshVAO, meshVBO, meshEBO;
    unsigned int meshVAO2, meshVBO2, meshEBO2;

    std::vector<glm::vec3> positions = std::vector<glm::vec3>(NumVertices);
    std::vector<glm::vec4> colors = std::vector<glm::vec4>(NumVertices);
    std::vector<glm::vec3> normals = std::vector<glm::vec3>(NumVertices);

    float histoDisplay[histoSize];

    glm::vec3 cameraPosition;
    glm::vec3 cameraRotation;
  };
/// [context]

  int deviceId = -1;

  for(int argI = 1; argI < argc; ++argI) {
    const std::string arg(argv[argI]);

    if(arg == "-help" || arg == "--help" || arg == "-h" || arg == "-v" || arg == "--version" || arg == "-version") {
      // Just let the initial lines display at the beginning of main
      return 0;
    }
    else if(arg.find("-gpu=") == 0) {
      if (pipeline) {
        std::cerr << "-gpu must be specified before pipeline argument" << std::endl;
        return -1;
      }
      deviceId = atoi(argv[argI] + 5);
    }
    else if(arg == "cpu") {
      if(!pipeline)
/// [pipeline]
        pipeline = new libfreenect2::CpuPacketPipeline();
/// [pipeline]
    }
    else if(arg == "gl") {
#ifdef LIBFREENECT2_WITH_OPENGL_SUPPORT
      if(!pipeline)
        pipeline = new libfreenect2::OpenGLPacketPipeline();
#else
      std::cout << "OpenGL pipeline is not supported!" << std::endl;
#endif
    }
    else if(arg == "cl") {
#ifdef LIBFREENECT2_WITH_OPENCL_SUPPORT
      if(!pipeline)
        pipeline = new libfreenect2::OpenCLPacketPipeline(deviceId);
#else
      std::cout << "OpenCL pipeline is not supported!" << std::endl;
#endif
    }
    else if(arg == "clkde") {
#ifdef LIBFREENECT2_WITH_OPENCL_SUPPORT
      if(!pipeline)
        pipeline = new libfreenect2::OpenCLKdePacketPipeline(deviceId);
#else
      std::cout << "OpenCL pipeline is not supported!" << std::endl;
#endif
    }
    else if(arg == "cuda") {
#ifdef LIBFREENECT2_WITH_CUDA_SUPPORT
      if(!pipeline)
        pipeline = new libfreenect2::CudaPacketPipeline(deviceId);
#else
      std::cout << "CUDA pipeline is not supported!" << std::endl;
#endif
    }
    else if(arg == "cudakde") {
#ifdef LIBFREENECT2_WITH_CUDA_SUPPORT
      if(!pipeline)
        pipeline = new libfreenect2::CudaKdePacketPipeline(deviceId);
#else
      std::cout << "CUDA pipeline is not supported!" << std::endl;
#endif
    }
    else {
      std::cout << "Unknown argument: " << arg << std::endl;
    }
  }

/// [discovery]
  int numDevices = freenect2.enumerateDevices();
  if(numDevices == 0) {
    std::cout << "no device connected!" << std::endl;
    return -1;
  }

  std::vector<kinect_device> devices(numDevices);
  for (int k = 0; k < numDevices; k++) {
    devices[k].serial = freenect2.getDeviceSerialNumber(k);
  }
/// [discovery]

  if(pipeline) {
/// [open]
    for (int k = 0; k < numDevices; k++) {
      devices[k].dev = freenect2.openDevice(devices[k].serial, pipeline);
    }
/// [open]
  }
  else {
    for (int k = 0; k < numDevices; k++) {
      devices[k].dev = freenect2.openDevice(devices[k].serial);
    }
  }

  if(devices[0].dev == 0) {
    std::cout << "failure opening device!" << std::endl;
    return -1;
  }

  devtopause = devices[0].dev;

  signal(SIGINT,sigint_handler);
#ifdef SIGUSR1
  signal(SIGUSR1, sigusr1_handler);
#endif
  protonect_shutdown = false;


  int types = libfreenect2::Frame::Color | libfreenect2::Frame::Depth;
  for (int k = 0; k < numDevices; k++) {
    /// [listeners]
    devices[k].listener = new libfreenect2::SyncMultiFrameListener(types);
    devices[k].dev->setColorFrameListener(devices[k].listener);
    devices[k].dev->setIrAndDepthFrameListener(devices[k].listener);
    
    /// [start]
    if (!devices[k].dev->start()) {
      std::cout << "failure starting device " << k << "!" << std::endl;
      return -1;
    }

    /// [registration setup]
    devices[k].registration = new libfreenect2::Registration(devices[k].dev->getIrCameraParams(), devices[k].dev->getColorCameraParams());
  }

  std::cout << "Devices:" << std::endl;
  for (int k = 0; k < numDevices; k++) {
    std::cout << k << std::endl;
    std::cout << "  device serial: " << devices[k].dev->getSerialNumber() << std::endl;
    std::cout << "  device firmware: " << devices[k].dev->getFirmwareVersion() << std::endl;
  }


/// [loop start]
  // mesh
  std::vector<unsigned int> meshIndices((M-1) * (N-1) * 2 * 3);

  int index = 0;
  for (int y = 0; y < M-1; y++) {
      for (int x = 0; x < N-1; x++) {
          unsigned int v = y*N + x;
          meshIndices[index]   = v;
          meshIndices[index+1] = v + 1;
          meshIndices[index+2] = v+N;

          meshIndices[index+3] = v + 1;
          meshIndices[index+4] = v+N + 1;
          meshIndices[index+5] = v+N;
          index += 6;
      }
  }

  for (int k = 0; k < numDevices; k++) {
    glGenVertexArrays(1, &devices[k].meshVAO);
    glGenBuffers(1, &devices[k].meshVBO);
    glGenBuffers(1, &devices[k].meshEBO);
  
    glBindVertexArray(devices[k].meshVAO);
  
    glBindBuffer(GL_ARRAY_BUFFER, devices[k].meshVBO);
    glBufferData(GL_ARRAY_BUFFER, devices[k].meshVertices.size() * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
  
    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 10 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 10 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // normal attribute
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 10 * sizeof(float), (void*)(7 * sizeof(float)));
    glEnableVertexAttribArray(2);
  
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, devices[k].meshEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, meshIndices.size() * sizeof(unsigned int), meshIndices.data(), GL_STATIC_DRAW);
  
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
  
  
    glGenVertexArrays(1, &devices[k].meshVAO2);
    glGenBuffers(1, &devices[k].meshVBO2);
    glGenBuffers(1, &devices[k].meshEBO2);
  
    glBindVertexArray(devices[k].meshVAO2);
  
    glBindBuffer(GL_ARRAY_BUFFER, devices[k].meshVBO2);
    glBufferData(GL_ARRAY_BUFFER, devices[k].meshVertices2.size() * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
  
    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 10 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 10 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // normal attribute
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 10 * sizeof(float), (void*)(7 * sizeof(float)));
    glEnableVertexAttribArray(2);
  
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, devices[k].meshEBO2);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, meshIndices.size() * sizeof(unsigned int), meshIndices.data(), GL_STATIC_DRAW);
  
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);


    //if (k == 0) {
    //  devices[k].cameraPosition = glm::vec3(-0.39f, 0.0f, 0.0f);
    //  devices[k].cameraRotation = glm::vec3(25.0f, 30.0f, 0.0f);
    //}
    //if (k == 1) {
    //  devices[k].cameraPosition = glm::vec3(0.39f, 0.0f, 0.0f);
    //  devices[k].cameraRotation = glm::vec3(25.0f, -25.0f, 0.0f);
    //}
  }
  


  // quads
  float quadVertices[] = {
    -0.5f, -0.5f,  0.0f,  1.0f, 0.0f, 0.0f, 1.0f,  0.0f, 0.0f, 1.0f,
     0.5f, -0.5f,  0.0f,  0.0f, 1.0f, 0.0f, 1.0f,  0.0f, 0.0f, 1.0f,
    -0.5f,  0.5f,  0.0f,  0.0f, 0.0f, 1.0f, 1.0f,  0.0f, 0.0f, 1.0f,
     0.5f,  0.5f,  0.0f,  1.0f, 1.0f, 0.0f, 1.0f,  0.0f, 0.0f, 1.0f
  };

  unsigned int quadIndices[] = {  
    0,  2,  1,   1,  2, 3
  };
  struct quad {
    unsigned int VAO, VBO, EBO;
  };
  std::vector<quad> quads(1000);
  for (size_t q = 0; q < quads.size(); q++) {
    glGenVertexArrays(1, &quads[q].VAO);
    glGenBuffers(1, &quads[q].VBO);
    glGenBuffers(1, &quads[q].EBO);

    glBindVertexArray(quads[q].VAO);

    glBindBuffer(GL_ARRAY_BUFFER, quads[q].VBO);
    glBufferData(GL_ARRAY_BUFFER, 4 * 10 * sizeof(float), quadVertices, GL_DYNAMIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 10 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 10 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // normal attribute
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 10 * sizeof(float), (void*)(7 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, quads[q].EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(quadIndices), quadIndices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
  }
  
  std::cout << "Threads: " << omp_get_max_threads() << "\n";

  float deltaTime = 0.0f;	// time between current frame and last frame
  float lastFrame = 0.0f;
  while(!glfwWindowShouldClose(window) && !protonect_shutdown) {
    float currentFrame = static_cast<float>(glfwGetTime());
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;

    for (int k = 0; k < numDevices; k++) {
      devices[k].newFrame = false;
      if (devices[k].listener->hasNewFrame() || devices[k].firstFrame) {
        devices[k].newFrame = true;
  
        if (devices[k].firstFrame)
          devices[k].firstFrame = false;                          // 10 seconds
        if (!devices[k].listener->waitForNewFrame(devices[k].frames, 10*1000)) {
          std::cout << "timeout!" << std::endl;
          return -1;
        }
        libfreenect2::Frame *rgb = devices[k].frames[libfreenect2::Frame::Color];
        libfreenect2::Frame *depth = devices[k].frames[libfreenect2::Frame::Depth];
  
        devices[k].registration->apply(rgb, depth, &devices[k].undistorted, &devices[k].registered);
  
        std::memcpy(devices[k].rgb.data, devices[k].registered.data, M * N * 4);
        cv::flip(devices[k].rgb, devices[k].rgb, 1);
        //std::memcpy(devices[k].rgbHD.data,   rgb->data, 1080 * 1920 * 4);
        //cv::flip(devices[k].rgbHD, devices[k].rgbHD, 1);
  
        cv::cvtColor(devices[k].rgb, devices[k].hsv, cv::COLOR_BGR2HSV);
        std::memcpy(devices[k].depth.data,   depth->data, M * N * 4);
        cv::flip(devices[k].depth, devices[k].depth, 1);
        //cv::blur(devices[k].depth, devices[k].depthSmooth, cv::Size( 10, 10 ), cv::Point(-1,-1) );
        cv::GaussianBlur( devices[k].depth, devices[k].depthSmooth, cv::Size( 7, 7 ), 0, 0 );
        //cv::medianBlur(devices[k].depth, devices[k].depth, 5);
        //std::memcpy(devices[k].depthSmooth.data,   devices[k].depth.data, M * N * 4);
        std::memcpy(devices[k].depth.data,   devices[k].depthSmooth.data, M * N * 4);

        if (k==0) {
          //cv::imwrite("rgb.jpg", devices[k].rgb);
          //cv::imwrite("depth.jpg", devices[k].depth);
        }

        devices[k].listener->release(devices[k].frames);
      }
    }


    float maxDeltaZ = 0.05f;
    float nonBlackThreshold = 0.0056f;
    float grayScaleThreshold = 0.06f;

    for (int k = 0; k < numDevices; k++) {
      int numQuads = 0;

      glm::mat4 T = glm::translate(glm::mat4(1.0f), devices[k].cameraPosition);
      glm::mat4 R = glm::rotate(glm::mat4(1.0f), glm::radians(-devices[k].cameraRotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
      R = glm::rotate(R, glm::radians(-devices[k].cameraRotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
      glm::mat4 transform = T * R;
      //glm::mat4 transform = glm::mat4(1.0f);

      
      if (devices[k].newFrame) {
        std::fill(devices[k].meshVertices.begin(), devices[k].meshVertices.end(), 0.0f);
        std::fill(devices[k].meshVertices2.begin(), devices[k].meshVertices2.end(), 0.0f);
  
        std::vector<int> histoMap(M*N, -1);
        std::vector<std::pair<unsigned int, unsigned int>> histo(histoSize, std::pair<unsigned int, unsigned int>(0, 0));
        for (int i = 0; i < histoSize; i++) {
          histo[i].first = i;
        }
        std::vector<std::vector<unsigned int>> histoPoints(histoSize, std::vector<unsigned int>());
  
        #pragma omp parallel for
        for (int i = 0; i < M*N; i++) {
          int m = i / N;
          int n = i % N; 
  
          float depth = devices[k].depth.at<float>(m, n)/1000.0f;
          uchar* pixel = devices[k].rgb.ptr<uchar>(m, n);
          
          glm::vec3 pos = glm::vec3((static_cast<float>(n) - cx) * depth / fx, (static_cast<float>(M-1 - m) - cy) * depth / fy, -depth);
          pos = glm::vec3(transform * glm::vec4(pos, 1.0f));
          glm::vec4 color = glm::vec4(*(pixel+2)/255.0f, *(pixel+1)/255.0f, *(pixel)/255.0f, 1.0f);
  
          glm::vec3 posOtherHorizontal = glm::vec3(0.0f);
          glm::vec3 posOtherVertical = glm::vec3(0.0f);
          if (depth == 0.0f || depth >= 2.5f) {
            color.w = 0.0f;
          }
          for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
              if (y == 0 && x == 0)
                continue;
              if (m+y < 0 || m+y >= M || n+x < 0 || n+x >= N)
                continue;
              float depthOther = devices[k].depth.at<float>(m+y, n+x)/1000.0f;
              if (abs(depth-depthOther) > maxDeltaZ) {
                color.w = 0.0f;
              }
  
              float depthSmoothOther = devices[k].depthSmooth.at<float>(m+y, n+x)/1000.0f;
              glm::vec3 posOther = glm::vec3((static_cast<float>(n+x) - cx) * depthSmoothOther / fx, (static_cast<float>(M-1 - (m+y)) - cy) * depthSmoothOther / fy, -depthSmoothOther);
              posOther = glm::vec3(transform * glm::vec4(posOther, 1.0f));
              if (y==0) {
                posOtherHorizontal = posOther;
              }
              if (x==0) {
                posOtherVertical = posOther;
              }
            }
          }
          float depthFlat = devices[k].depthSmooth.at<float>(m, n)/1000.0f;
          glm::vec3 posFlat = glm::vec3((static_cast<float>(n) - cx) * depthFlat / fx, (static_cast<float>(M-1 - m) - cy) * depthFlat / fy, -depthFlat);
          posFlat = glm::vec3(transform * glm::vec4(posFlat, 1.0f));
          glm::vec3 a = posOtherHorizontal - posFlat;
          glm::vec3 b = posOtherVertical - posFlat;
          glm::vec3 normal = glm::normalize(glm::cross(b, a));
          
  
          if (color.w > 0.1f && (color.x > nonBlackThreshold || color.y > nonBlackThreshold || color.z > nonBlackThreshold)) {
              unsigned int pixelHisto;
              if (abs(color.x - color.y) < grayScaleThreshold && abs(color.y - color.z) < grayScaleThreshold && abs(color.z - color.x) < grayScaleThreshold) {
                int pixelGray = 180 + static_cast<int>(color.x*179.0f);
                pixelHisto = pixelGray;
              }
              else {
                uchar* pixelHSV = devices[k].hsv.ptr<uchar>(m, n);
                pixelHisto = *pixelHSV;
              }
  
              histoMap[i] = pixelHisto;
          }
          
          int index = m*N + n;
          int index2 = (m*N + n)*10;
          devices[k].meshVertices[index2 + 0] = pos.x;
          devices[k].meshVertices[index2 + 1] = pos.y;
          devices[k].meshVertices[index2 + 2] = pos.z;
          devices[k].meshVertices[index2 + 3] = color.x;
          devices[k].meshVertices[index2 + 4] = color.y;
          devices[k].meshVertices[index2 + 5] = color.z;
          devices[k].meshVertices[index2 + 6] = color.w;
          devices[k].meshVertices[index2 + 7] = normal.x;
          devices[k].meshVertices[index2 + 8] = normal.y;
          devices[k].meshVertices[index2 + 9] = normal.z;
  
          devices[k].meshVertices2[index2 + 0] = pos.x;
          devices[k].meshVertices2[index2 + 1] = pos.y;
          devices[k].meshVertices2[index2 + 2] = pos.z;
          devices[k].meshVertices2[index2 + 7] = normal.x;
          devices[k].meshVertices2[index2 + 8] = normal.y;
          devices[k].meshVertices2[index2 + 9] = normal.z;
  
          devices[k].positions[index] = pos;
          devices[k].colors[index] = color;
          devices[k].normals[index] = normal;
        }
  
        for (int i = 0; i < M*N; i++) {
          int pixelHisto = histoMap[i];
          if (pixelHisto != -1) {
              histo[pixelHisto].second += 1;
              histoPoints[pixelHisto].push_back(i);
          }
        }
        
        
        unsigned int maxCount = 1;
        for (int i = 0; i < histoSize; i++) {
          devices[k].histoDisplay[i] = static_cast<float>(histo[i].second);
          if (histo[i].second > maxCount) {
            maxCount = histo[i].second;
          }
        }
        float maxCountf = static_cast<float>(maxCount);
        for (int i = 0; i < histoSize; i++) {
          devices[k].histoDisplay[i] /= maxCountf;
        }
  
  
        
        auto hueComp = [](std::pair<unsigned int, unsigned int> a, std::pair<unsigned int, unsigned int> b) {
          return a.first < b.first;
        };
        auto countComp = [](std::pair<unsigned int, unsigned int> a, std::pair<unsigned int, unsigned int> b) {
          return a.second > b.second;
        };
  
        
        // parameters
        int maxAttempts = 100;
        int radius = 1;
        unsigned int planeThreshold = 5;
        float stdDev = 0.0f;
        float distThreshold = 0.05f;
        float dotThreshold = cosf(Deg2Rad * 15.0f);
        float dotThresholdRelaxed = cosf(Deg2Rad * 45.0f);
  
        bool foundPlane = true;
        int numPlanes = 0;
        while (foundPlane) {
          foundPlane = false;
          std::sort(histo.begin(), histo.end(), countComp);
          for (int r = 0; r < histoSize; r++) {
            if (histo[r].second < planeThreshold) {
              break;
            }
            unsigned int hue = static_cast<unsigned int>(histo[r].first);
            std::sort(histo.begin(), histo.end(), hueComp);
  
            int left = 0;
            int right = 0;
            if (hue < 180) {
              for (int x = -1; x >= -89; x--) {
                int hue_x = hue + x;
                if (hue_x < 0) {
                  hue_x += 180;
                }
                if (histo[hue_x].second == 0)
                  break;
                left = x;
              }
              for (int x = 1; x <= 90; x++) {
                int hue_x = hue + x;
                if (hue_x >= 180) {
                  hue_x -= 180;
                }
                if (histo[hue_x].second == 0)
                  break;
                right = x;
              }
            }
            else {
              for (int x = -1; (hue+x) >= 0; x--) {
                int hue_x = hue + x;
                if (histo[hue_x].second == 0)
                  break;
                left = x;
              }
              for (int x = 1; (hue+x) < histoSize; x++) {
                int hue_x = hue + x;
                if (histo[hue_x].second == 0)
                  break;
                right = x;
              }
            }
  
  
            std::vector<std::pair<glm::u32mat3x2, unsigned int>> possiblePlanes(maxAttempts, std::pair<glm::u32mat3x2, unsigned int>(glm::u32mat3x2(0), 0));
             
            bool existsPlane = false;
            #pragma omp parallel
            {
              int thread_id = omp_get_thread_num();
  
              std::random_device rd{};
              std::mt19937 gen{rd()};
              gen.seed(thread_id);
              std::normal_distribution<float> normalDistribution{0.0f, stdDev};
              srand(thread_id);
  
              int numThreads = omp_get_num_threads();
              int chunk = std::ceil(static_cast<float>(maxAttempts)/static_cast<float>(numThreads));
              int boundary = thread_id == numThreads-1 ? maxAttempts : (thread_id+1)*chunk;
              for (int attempt = thread_id*chunk; attempt < boundary; attempt++) {  
                unsigned int rd_huesAndIndices[6];
                unsigned int planeIndices[3];
                for (int i = 0; i < 3; i++) {
                  int rd_hue = static_cast<int>(hue) + std::min(right, std::max(left, static_cast<int>(normalDistribution(gen))));
                  rd_hue = hue < 180 ? (rd_hue < 0 ? rd_hue+=180 : (rd_hue >= 180 ? rd_hue-=180 : rd_hue)) : rd_hue;
                  int rd_num = rand() % histoPoints[rd_hue].size();
                  unsigned int vertIndex = histoPoints[rd_hue][rd_num];
                  
                  rd_huesAndIndices[i*2 + 0] = rd_hue;
                  rd_huesAndIndices[i*2 + 1] = vertIndex;
                  planeIndices[i] = vertIndex;
                }
                if (planeIndices[0] == planeIndices[1] || planeIndices[0] == planeIndices[2] || planeIndices[1] == planeIndices[2]) {
                  continue;
                }
                
                
                glm::vec3 a = devices[k].positions[planeIndices[1]] - devices[k].positions[planeIndices[0]];
                glm::vec3 b = devices[k].positions[planeIndices[2]] - devices[k].positions[planeIndices[0]];
                glm::vec3 normal = glm::normalize(glm::cross(a, b));
                //std::cout << normal.x << " " << normal.y << " " << normal.z << "\n";
                float d = glm::dot(normal, -devices[k].positions[planeIndices[0]]);
        
                unsigned int count = 0;
                //#pragma omp parallel for reduction(+:count)
                for (int x = -radius; x <= radius; x++) {
                  if (x < left || x > right) {
                    continue;
                  }
                  int hue_x = hue + x;
                  hue_x = hue < 180 ? (hue_x < 0 ? hue_x+=180 : (hue_x >= 180 ? hue_x-=180 : hue_x)) : hue_x;
  
                  for (int i = 0; i < histoPoints[hue_x].size(); i++) {
                    unsigned int vertIndex = histoPoints[hue_x][i];
                    
                    float D = abs(glm::dot(normal, devices[k].positions[vertIndex]) + d);
                    if (D < distThreshold && abs(glm::dot(normal, devices[k].normals[vertIndex])) > dotThreshold) {
                      count++;
                    }
                  }
                }
    
                if (count >= planeThreshold) {
                  existsPlane = true;
                  possiblePlanes[attempt] = std::pair<glm::u32mat3x2, unsigned int>(glm::u32mat3x2(rd_huesAndIndices[0], rd_huesAndIndices[1], rd_huesAndIndices[2], rd_huesAndIndices[3], rd_huesAndIndices[4], rd_huesAndIndices[5]), count);
                }
              }
            }
            
            
            // draw Quad
            if (!existsPlane) {
              continue;
            }
            else {
              unsigned int maxPointIndex = 0;
              unsigned int maxPointCount = possiblePlanes[0].second;
              for (int i = 1; i < maxAttempts; i++) {
                if (possiblePlanes[i].second > maxPointCount) {
                  maxPointIndex = i;
                  maxPointCount = possiblePlanes[i].second;
                }
              }
  
  
              unsigned int rd_hue1 = possiblePlanes[maxPointIndex].first[0][0];
              unsigned int vert1Index = possiblePlanes[maxPointIndex].first[0][1];
              unsigned int rd_hue2 = possiblePlanes[maxPointIndex].first[1][0];
              unsigned int vert2Index = possiblePlanes[maxPointIndex].first[1][1];
              unsigned int rd_hue3 = possiblePlanes[maxPointIndex].first[2][0];
              unsigned int vert3Index = possiblePlanes[maxPointIndex].first[2][1];
  
              histo[rd_hue1].second -= 1;
              histo[rd_hue2].second -= 1;
              histo[rd_hue3].second -= 1;
              histoPoints[rd_hue1].erase(std::lower_bound(histoPoints[rd_hue1].begin(), histoPoints[rd_hue1].end(), vert1Index));
              histoPoints[rd_hue2].erase(std::lower_bound(histoPoints[rd_hue2].begin(), histoPoints[rd_hue2].end(), vert2Index));
              histoPoints[rd_hue3].erase(std::lower_bound(histoPoints[rd_hue3].begin(), histoPoints[rd_hue3].end(), vert3Index));
  
  
              std::vector<unsigned int> planePointsIndices;
              planePointsIndices.push_back(vert1Index);
              planePointsIndices.push_back(vert2Index);
              planePointsIndices.push_back(vert3Index);
  
              glm::vec3 a = devices[k].positions[vert2Index] - devices[k].positions[vert1Index];
              glm::vec3 b = devices[k].positions[vert3Index] - devices[k].positions[vert1Index];
              glm::vec3 normal = glm::normalize(glm::cross(a, b));
              float d = glm::dot(normal, -devices[k].positions[vert1Index]);
      
              for (int hue_x = 0; hue_x < histoSize; hue_x++) {
                int count = 0;
                std::vector<unsigned int> remainingVertIndices;
                for (int i = 0; i < histoPoints[hue_x].size(); i++) {
                  unsigned int vertIndex = histoPoints[hue_x][i];
                            
                  float D = abs(glm::dot(normal, devices[k].positions[vertIndex]) + d);
                  if (D < distThreshold && abs(glm::dot(normal, devices[k].normals[vertIndex])) > dotThresholdRelaxed) {
                    planePointsIndices.push_back(vertIndex);
                    count++;
                  }
                  else {
                    remainingVertIndices.push_back(vertIndex);
                  }
                }
                histo[hue_x].second -= count;
                histoPoints[hue_x] = remainingVertIndices;
              }
  
    
              glm::vec4 planeColor = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
              /*
              for (int i = 0; i < planePointsIndices.size(); i++) {
                unsigned int vertIndex = planePointsIndices[i];
    
                planeColor.x += colors[vertIndex].x;
                planeColor.y += colors[vertIndex].y;
                planeColor.z += colors[vertIndex].z;
                planeColor.w += colors[vertIndex].w;
              }
              planeColor.x /= planePointsIndices.size();
              planeColor.y /= planePointsIndices.size();
              planeColor.z /= planePointsIndices.size();
              planeColor.w /= planePointsIndices.size();
              */
              
              planeColor = glm::vec4((numPlanes >> 0) & 1, (numPlanes >> 1) & 1, (numPlanes >> 2) & 1, 1.0f);
              for (int i = 0; i < planePointsIndices.size(); i++) {
                unsigned int vertIndex = planePointsIndices[i];
    
                devices[k].meshVertices2[vertIndex*10 + 3] = planeColor.x;
                devices[k].meshVertices2[vertIndex*10 + 4] = planeColor.y;
                devices[k].meshVertices2[vertIndex*10 + 5] = planeColor.z;
                devices[k].meshVertices2[vertIndex*10 + 6] = planeColor.w;
              }
  
              foundPlane = true;
              numPlanes++;
              break;
            }
          }
        }
        
  
        glBindVertexArray(devices[k].meshVAO);
        glBindBuffer(GL_ARRAY_BUFFER, devices[k].meshVBO);
        glBufferData(GL_ARRAY_BUFFER, devices[k].meshVertices.size() * sizeof(float), devices[k].meshVertices.data(), GL_DYNAMIC_DRAW);
  
        glBindVertexArray(devices[k].meshVAO2);
        glBindBuffer(GL_ARRAY_BUFFER, devices[k].meshVBO2);
        glBufferData(GL_ARRAY_BUFFER, devices[k].meshVertices2.size() * sizeof(float), devices[k].meshVertices2.data(), GL_DYNAMIC_DRAW);
  
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
      }
    }



    //glViewport(0, 0, 1280, 720);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glm::mat4 P = glm::perspective(glm::radians(89.0f), 1280.0f / 720.0f, 0.01f, 1000.0f);

    // Update Camera
    const auto& io = ImGui::GetIO();

    if (!io.WantCaptureKeyboard) {
      // movement
      float multiplier = 1.0f;
      if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
          multiplier = 30.0f;
      }
              
      int xInput = 0;
      if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
          xInput += 1;
      }
      if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
          xInput -= 1;
      }
      if (xInput > 0) {
          ZPos -= multiplier * MovementSpeed * deltaTime;
      }
      else if (xInput < 0) {
          ZPos += multiplier * MovementSpeed * deltaTime;
      }
    }

    if (!io.WantCaptureMouse) {
      // rotation
      if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
        cameraRotation.y += RotationSpeed * DeltaX;
        if (cameraRotation.y < 0.0f)
            cameraRotation.y += 360.0f;
        else if (cameraRotation.y > 360.0f)
            cameraRotation.y -= 360.0f;
  
        cameraRotation.x += RotationSpeed * DeltaY;
        if (cameraRotation.x < -89.0f)
            cameraRotation.x = -89.0f;
        else if (cameraRotation.x > 89.0f)
            cameraRotation.x = 89.0f;
  
        DeltaX = 0.0f;
        DeltaY = 0.0f;
      }


      // zoom
      ZoomDistance += YOffset * ZoomSpeed;
      YOffset = 0.0f;
      if (ZoomDistance < 0.2f) {
          ZoomDistance = 0.2f;
      }
      else if (ZoomDistance > 20.0f) {
          ZoomDistance = 20.0f;
      }
    }


    // Update Camera Transformation Matrices
    glm::vec4 cameraPos = glm::vec4(0.0f, 0.0f, ZoomDistance, 1.0f);
        
    R = glm::rotate(glm::mat4(1.0f), glm::radians(-cameraRotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
    R = glm::rotate(R, glm::radians(-cameraRotation.x), glm::vec3(1.0f, 0.0f, 0.0f));

    cameraPos = R * cameraPos;
    Position = glm::vec3(cameraPos.x, cameraPos.y, cameraPos.z + ZPos);
    Front = glm::normalize(glm::vec3(0.0f, 0.0f, ZPos) - Position);

    glm::mat4 V = glm::lookAt(Position, Position + Front, Up);
    glm::mat4 VP = P * V;


    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    //ImGui::ShowDemoWindow(0);
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
    ImGui::NewLine();

    //static float x = 0.0f;
    //ImGui::DragFloat("drag float", &x, 0.005f, -5.0f, 5.0f);
    for (int k = 0; k < numDevices; k++) {
      std::string cameraPosString = "CameraPos-" + std::to_string(k);
      std::string cameraRotString = "CameraRot-" + std::to_string(k);
      ImGui::DragFloat3(cameraPosString.c_str(), &devices[k].cameraPosition.x, 0.01f, -0.5f, 0.5f);
      ImGui::DragFloat3(cameraRotString.c_str(), &devices[k].cameraRotation.x, 1.0f, -90.0f, 90.0f);

      std::string histoString = "Histogram-" + std::to_string(k);
      ImGui::PlotHistogram(histoString.c_str(), devices[k].histoDisplay, histoSize, 0, NULL, 0.0f, 1.0f, ImVec2(0, 80.0f));
    }


    
    meshShader.use();
    meshShader.setMat4("WVP", VP);
    //meshShader.setMat4("WVP", VP * glm::translate(glm::mat4(1.0f), glm::vec3(-2.0f, 0.0f, 0.0f)));
    glBindVertexArray(devices[0].meshVAO2);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, devices[0].meshEBO2);
    glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(meshIndices.size()), GL_UNSIGNED_INT, 0);

    meshShader.setMat4("WVP", VP);
    //meshShader.setMat4("WVP", VP * glm::translate(glm::mat4(1.0f), glm::vec3(2.0f, 0.0f, 0.0f)));
    glBindVertexArray(devices[1].meshVAO2);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, devices[1].meshEBO2);
    glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(meshIndices.size()), GL_UNSIGNED_INT, 0);

    //meshShader.use();
    ////meshShader.setMat4("WVP", VP);
    //meshShader.setMat4("WVP", VP * glm::translate(glm::mat4(1.0f), glm::vec3(2.0f, 0.0f, 0.0f)));
    //for (int i = 0; i < numQuads; i++) {
    //  glBindVertexArray(quads[i].VAO);
    //  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, quads[i].EBO);
    //  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    //}
    

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    for (int k = 0; k < numDevices; k++) {
      std::string rgbString = "rgb-" + std::to_string(k);
      std::string depthString = "depth-" + std::to_string(k);
      std::string depthSmoothString = "depthSmooth-" + std::to_string(k);
      std::string rgbHDString = "rgbHD-" + std::to_string(k);
      std::string hsvString = "hsv-" + std::to_string(k);
      std::string hsvHueString = "hsvHue-" + std::to_string(k);

      cv::imshow(rgbString, devices[k].rgb);
      cv::imshow(depthString, devices[k].depth/4500.0f);
      cv::imshow(depthSmoothString, devices[k].depthSmooth/4500.0f);
      
      //cv::imshow(rgbHDString, devices[k].rgbHD);
      //cv::imshow(hsvString, devices[k].hsv);
      //std::vector<cv::Mat> channels;
      //cv::split(devices[k].hsv, channels);
      //cv::imshow(hsvHueString, channels[0]);
    }
    
    int key = cv::waitKey(1);
    if (key == 'q')
      return -1;

/// [loop end]
    //listener.release(frames);
    /** libfreenect2::this_thread::sleep_for(libfreenect2::chrono::milliseconds(100)); */

    glfwSwapBuffers(window);
    glfwPollEvents();
  }
/// [loop end]

  // TODO: restarting ir stream doesn't work!
  // TODO: bad things will happen, if frame listeners are freed before dev->stop() :(
/// [stop]
  for (int k = 0; k < numDevices; k++) {
    delete devices[k].registration;

    devices[k].dev->stop();
    devices[k].dev->close();

    delete devices[k].listener;
  }
/// [stop]


  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  //glfwTerminate();


  return 0;
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    //glViewport(0, 0, width, height);
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (FirstMouse)
    {
        LastX = xpos;
        LastY = ypos;
        FirstMouse = false;
    }

    DeltaX = xpos - LastX;
    DeltaY = ypos - LastY; // reversed since y-coordinates go from bottom to top

    LastX = xpos;
    LastY = ypos;
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    YOffset = static_cast<float>(-yoffset);
}