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
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"

bool protonect_shutdown = false; ///< Whether the running application should shut down.

void sigint_handler(int s)
{
  protonect_shutdown = true;
}

bool protonect_paused = false;
libfreenect2::Freenect2Device *devtopause;

//Doing non-trivial things in signal handler is bad. If you want to pause,
//do it in another thread.
//Though libusb operations are generally thread safe, I cannot guarantee
//everything above is thread safe when calling start()/stop() while
//waitForNewFrame().
void sigusr1_handler(int s)
{
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
class MyFileLogger: public libfreenect2::Logger
{
private:
  std::ofstream logfile_;
public:
  MyFileLogger(const char *filename)
  {
    if (filename)
      logfile_.open(filename);
    level_ = Debug;
  }
  bool good()
  {
    return logfile_.is_open() && logfile_.good();
  }
  virtual void log(Level level, const std::string &message)
  {
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
 * - <number> Serial number of the device to open.
 */



void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

// Camera Controls
float ZPos = 0.0f;
float MovementSpeed = 1.0f;

glm::vec3 cameraRotation(25.0f, 315.0f, 0.0f);
float RotationSpeed = 0.5f;
float LastX = 0.0f;
float LastY = 0.0f;
float DeltaX = 0.0f;
float DeltaY = 0.0f;
bool FirstMouse = true;

float YOffset = 0.0f;
float ZoomDistance = 2.0f;
float ZoomSpeed = 0.2f;

// Camera
glm::vec3 Up = glm::vec3(0.0f, 1.0f, 0.0f);
glm::vec3 Position = glm::vec3(0.0f, 0.0f, ZoomDistance);
glm::vec3 Front = glm::vec3(0.0f, 0.0f, -1.0f);
glm::mat4 T(1.0f);
glm::mat4 R(1.0f);


int main(int argc, char *argv[])
/// [main]
{
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
    if (window == NULL)
    {
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

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
  }
  

  glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
  glEnable(GL_DEPTH_TEST);
  //glPolygonOffset(1, 1);
  //glEnable(GL_BLEND);
  //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  Shader voxelShader = Shader("examples/shaders/voxel.vert", "examples/shaders/voxel.frag");
  Shader meshShader = Shader("examples/shaders/mesh.vert", "examples/shaders/mesh.frag");


  std::string program_path(argv[0]);
  std::cerr << "Version: " << LIBFREENECT2_VERSION << std::endl;
  std::cerr << "Environment variables: LOGFILE=<protonect.log>" << std::endl;
  std::cerr << "Usage: " << program_path << " [-gpu=<id>] [gl | cl | clkde | cuda | cudakde | cpu] [<device serial>]" << std::endl;
  std::cerr << "        [-norgb | -nodepth] [-help] [-version]" << std::endl;
  std::cerr << "        [-frames <number of frames to process>]" << std::endl;
  std::cerr << "To pause and unpause: pkill -USR1 Protonect" << std::endl;
  size_t executable_name_idx = program_path.rfind("Protonect");

  std::string binpath = "/";

  if(executable_name_idx != std::string::npos)
  {
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
  libfreenect2::Freenect2Device *dev = 0;
  libfreenect2::PacketPipeline *pipeline = 0;
/// [context]

  std::string serial = "";

  int deviceId = -1;
  size_t framemax = -1;

  for(int argI = 1; argI < argc; ++argI)
  {
    const std::string arg(argv[argI]);

    if(arg == "-help" || arg == "--help" || arg == "-h" || arg == "-v" || arg == "--version" || arg == "-version")
    {
      // Just let the initial lines display at the beginning of main
      return 0;
    }
    else if(arg.find("-gpu=") == 0)
    {
      if (pipeline)
      {
        std::cerr << "-gpu must be specified before pipeline argument" << std::endl;
        return -1;
      }
      deviceId = atoi(argv[argI] + 5);
    }
    else if(arg == "cpu")
    {
      if(!pipeline)
/// [pipeline]
        pipeline = new libfreenect2::CpuPacketPipeline();
/// [pipeline]
    }
    else if(arg == "gl")
    {
#ifdef LIBFREENECT2_WITH_OPENGL_SUPPORT
      if(!pipeline)
        pipeline = new libfreenect2::OpenGLPacketPipeline();
#else
      std::cout << "OpenGL pipeline is not supported!" << std::endl;
#endif
    }
    else if(arg == "cl")
    {
#ifdef LIBFREENECT2_WITH_OPENCL_SUPPORT
      if(!pipeline)
        pipeline = new libfreenect2::OpenCLPacketPipeline(deviceId);
#else
      std::cout << "OpenCL pipeline is not supported!" << std::endl;
#endif
    }
    else if(arg == "clkde")
    {
#ifdef LIBFREENECT2_WITH_OPENCL_SUPPORT
      if(!pipeline)
        pipeline = new libfreenect2::OpenCLKdePacketPipeline(deviceId);
#else
      std::cout << "OpenCL pipeline is not supported!" << std::endl;
#endif
    }
    else if(arg == "cuda")
    {
#ifdef LIBFREENECT2_WITH_CUDA_SUPPORT
      if(!pipeline)
        pipeline = new libfreenect2::CudaPacketPipeline(deviceId);
#else
      std::cout << "CUDA pipeline is not supported!" << std::endl;
#endif
    }
    else if(arg == "cudakde")
    {
#ifdef LIBFREENECT2_WITH_CUDA_SUPPORT
      if(!pipeline)
        pipeline = new libfreenect2::CudaKdePacketPipeline(deviceId);
#else
      std::cout << "CUDA pipeline is not supported!" << std::endl;
#endif
    }
    else if(arg.find_first_not_of("0123456789") == std::string::npos) //check if parameter could be a serial number
    {
      serial = arg;
    }
    else if(arg == "-frames")
    {
      ++argI;
      framemax = strtol(argv[argI], NULL, 0);
      if (framemax == 0) {
        std::cerr << "invalid frame count '" << argv[argI] << "'" << std::endl;
        return -1;
      }
    }
    else
    {
      std::cout << "Unknown argument: " << arg << std::endl;
    }
  }

/// [discovery]
  if(freenect2.enumerateDevices() == 0)
  {
    std::cout << "no device connected!" << std::endl;
    return -1;
  }

  if (serial == "")
  {
    serial = freenect2.getDefaultDeviceSerialNumber();
    //std::string serial2 = freenect2.getDefaultDeviceSerialNumber();
    //std::cout << "Serial: " << serial << "\n";
    //std::cout << "Serial2: " << serial2 << "\n";
  }
/// [discovery]

  if(pipeline)
  {
/// [open]
    dev = freenect2.openDevice(serial, pipeline);
/// [open]
  }
  else
  {
    dev = freenect2.openDevice(serial);
  }

  if(dev == 0)
  {
    std::cout << "failure opening device!" << std::endl;
    return -1;
  }

  devtopause = dev;

  signal(SIGINT,sigint_handler);
#ifdef SIGUSR1
  signal(SIGUSR1, sigusr1_handler);
#endif
  protonect_shutdown = false;

/// [listeners]
  int types = libfreenect2::Frame::Color | libfreenect2::Frame::Ir | libfreenect2::Frame::Depth;
  libfreenect2::SyncMultiFrameListener listener(types);
  libfreenect2::FrameMap frames;

  dev->setColorFrameListener(&listener);
  dev->setIrAndDepthFrameListener(&listener);
/// [listeners]

/// [start]
  if (!dev->start())
    return -1;

  std::cout << "device serial: " << dev->getSerialNumber() << std::endl;
  std::cout << "device firmware: " << dev->getFirmwareVersion() << std::endl;
/// [start]

/// [registration setup]
  libfreenect2::Registration* registration = new libfreenect2::Registration(dev->getIrCameraParams(), dev->getColorCameraParams());
  libfreenect2::Frame undistorted(512, 424, 4), registered(512, 424, 4);
/// [registration setup]

  size_t framecount = 0;


/// [loop start]
  const int M = 424;
  const int N = 512;
  cv::Mat cvRGB(M, N, CV_8UC4);
  cv::Mat hdRGB(1080, 1920, CV_8UC4);

  cv::Mat cvHSV(M, N, CV_8UC3);
  cv::Mat cvDepth(M, N, CV_32FC1);
  float numPoints = M * N;
  float ratio = -static_cast<float>(N) / static_cast<float>(M);

  const float PI = 3.1415926535897932385f;
  const float Deg2Rad = PI/180.0f;
  float vFOV = 60.0f;
  float vDegreesPerPixel = vFOV/static_cast<float>(M);
  float hFOV = 70.6f;
  float hDegreesPerPixel = hFOV/static_cast<float>(N);

  
  // mesh
  int numVertices = M*N;
  std::vector<float> meshVertices(numVertices*7);
  std::vector<float> meshVertices2(numVertices*7);

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

  unsigned int meshVAO, meshVBO, meshEBO;
  glGenVertexArrays(1, &meshVAO);
  glGenBuffers(1, &meshVBO);
  glGenBuffers(1, &meshEBO);

  glBindVertexArray(meshVAO);

  glBindBuffer(GL_ARRAY_BUFFER, meshVBO);
  glBufferData(GL_ARRAY_BUFFER, meshVertices.size() * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

  // position attribute
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);
  // color attribute
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, meshEBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, meshIndices.size() * sizeof(unsigned int), meshIndices.data(), GL_STATIC_DRAW);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);


  unsigned int meshVAO2, meshVBO2, meshEBO2;
  glGenVertexArrays(1, &meshVAO2);
  glGenBuffers(1, &meshVBO2);
  glGenBuffers(1, &meshEBO2);

  glBindVertexArray(meshVAO2);

  glBindBuffer(GL_ARRAY_BUFFER, meshVBO2);
  glBufferData(GL_ARRAY_BUFFER, meshVertices2.size() * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

  // position attribute
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);
  // color attribute
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, meshEBO2);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, meshIndices.size() * sizeof(unsigned int), meshIndices.data(), GL_STATIC_DRAW);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);


  // quads
  float quadVertices[] = {
    -0.5f, -0.5f,  0.0f,  1.0f, 0.0f, 0.0f, 1.0f,
     0.5f, -0.5f,  0.0f,  0.0f, 1.0f, 0.0f, 1.0f,
    -0.5f,  0.5f,  0.0f,  0.0f, 0.0f, 1.0f, 1.0f,
     0.5f,  0.5f,  0.0f,  1.0f, 1.0f, 0.0f, 1.0f
  };

  unsigned int quadIndices[] = {  
    0,  2,  1,   1,  2, 3
  };
  struct quad {
    unsigned int VAO, VBO, EBO;
  };
  std::vector<quad> quads(1000);
  for (size_t i = 0; i < quads.size(); i++) {
    glGenVertexArrays(1, &quads[i].VAO);
    glGenBuffers(1, &quads[i].VBO);
    glGenBuffers(1, &quads[i].EBO);

    glBindVertexArray(quads[i].VAO);

    glBindBuffer(GL_ARRAY_BUFFER, quads[i].VBO);
    glBufferData(GL_ARRAY_BUFFER, 4 * 7 * sizeof(float), quadVertices, GL_DYNAMIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, quads[i].EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(quadIndices), quadIndices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
  }
  

  float deltaTime = 0.0f;	// time between current frame and last frame
  float lastFrame = 0.0f;
  bool firstFrame = true;
  while(!glfwWindowShouldClose(window) && !protonect_shutdown && (framemax == (size_t)-1 || framecount < framemax))
  {
    float currentFrame = static_cast<float>(glfwGetTime());
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;

    //glViewport(0, 0, 1280, 720);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    

    glm::mat4 P = glm::perspective(glm::radians(89.0f), 1280.0f / 720.0f, 0.01f, 1000.0f);

    // Update Camera
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


    // Update Camera Transformation Matrices
    glm::vec4 cameraPos = glm::vec4(0.0f, 0.0f, ZoomDistance, 1.0f);
        
    R = glm::rotate(glm::mat4(1.0f), glm::radians(-cameraRotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
    R = glm::rotate(R, glm::radians(-cameraRotation.x), glm::vec3(1.0f, 0.0f, 0.0f));

    cameraPos = R * cameraPos;
    Position = glm::vec3(cameraPos.x, cameraPos.y, cameraPos.z + ZPos);
    Front = glm::normalize(glm::vec3(0.0f, 0.0f, ZPos) - Position);

    glm::mat4 V = glm::lookAt(Position, Position + Front, Up);
    glm::mat4 VP = P * V;


    int numQuads = 0;
    if (listener.hasNewFrame() || firstFrame) {
      if (firstFrame)
        firstFrame = false;
      if (!listener.waitForNewFrame(frames, 10*1000)) // 10 sconds
      {
        std::cout << "timeout!" << std::endl;
        return -1;
      }
      libfreenect2::Frame *rgb = frames[libfreenect2::Frame::Color];
      libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];
  /// [loop start]

  /// [registration]
      registration->apply(rgb, depth, &undistorted, &registered);

      framecount++;

      std::fill(meshVertices.begin(), meshVertices.end(), 0.0f);
      std::fill(meshVertices2.begin(), meshVertices2.end(), 0.0f);
      std::memcpy(cvRGB.data, registered.data, depth->height * depth->width * 4);
      cv::flip(cvRGB, cvRGB, 1);
      //std::memcpy(hdRGB.data,   rgb->data, rgb->height * rgb->width * 4);
      //cv::flip(hdRGB, hdRGB, 1);

      cv::cvtColor(cvRGB, cvHSV, cv::COLOR_RGB2HSV);
      std::memcpy(cvDepth.data,   depth->data, depth->height * depth->width * 4);
      cv::flip(cvDepth, cvDepth, 1);

      std::vector<std::pair<unsigned int, unsigned int>> histo(180, std::pair<unsigned int, unsigned int>(0, 0));
      for (int i = 0; i < 180; i++) {
        histo[i].first = i;
      }
      std::vector<std::vector<unsigned int>> histoPoints(180, std::vector<unsigned int>());

      float maxDeltaZ = 0.05f;
      for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
          float depth = cvDepth.at<float>(m, n)/1000.0f;
          uchar* pixel = cvRGB.ptr<uchar>(m, n);
          
          glm::vec3 pos = glm::vec3(tanf(Deg2Rad * (static_cast<float>(n) - N/2.0f)*hDegreesPerPixel) * depth, tanf(Deg2Rad * (static_cast<float>(M-1 - m) - M/2.0f)*vDegreesPerPixel) * depth, -depth);
          glm::vec4 color = glm::vec4(*(pixel+2)/255.0f, *(pixel+1)/255.0f, *(pixel)/255.0f, 1.0f);

          if (depth == 0.0f) {
            color.w = 0.0f;
          }
          else {
            bool exitLoop = false;
            for (int y = -1; y <= 1; y++) {
              for (int x = -1; x <= 1; x++) {
                if (y == 0 && x == 0)
                  continue;
                if (m+y < 0 || m+y >= M || n+x < 0 || n+x >= N)
                  continue;
                float depthOther = cvDepth.at<float>(m+y, n+x)/1000.0f;

                if (abs(depth-depthOther) > maxDeltaZ) {
                  color.w = 0.0f;
                  exitLoop = true;
                  break;
                }
              }
              if (exitLoop)
                break;
            }
          }
          

          if (color.w > 0.1f) {
              uchar* pixelHSV = cvHSV.ptr<uchar>(m, n);
              histo[*pixelHSV].second += 1;

              histoPoints[*pixelHSV].push_back((m*N + n)*7);
          }
          
          int index = (m*N + n)*7;
          meshVertices[index + 0] = pos.x;
          meshVertices[index + 1] = pos.y;
          meshVertices[index + 2] = pos.z;
          meshVertices[index + 3] = color.x;
          meshVertices[index + 4] = color.y;
          meshVertices[index + 5] = color.z;
          meshVertices[index + 6] = color.w;

          meshVertices2[index + 0] = pos.x;
          meshVertices2[index + 1] = pos.y;
          meshVertices2[index + 2] = pos.z;
        }
      }
      //std::memcpy(meshVertices2.data(), meshVertices.data(), meshVertices.size() * sizeof(float));


      
      auto hueComp = [](std::pair<unsigned int, unsigned int> a, std::pair<unsigned int, unsigned int> b) {
        return a.first < b.first;
      };

      auto countComp = [](std::pair<unsigned int, unsigned int> a, std::pair<unsigned int, unsigned int> b) {
        return a.second > b.second;
      };
      
      /*
      while (true) {
        std::sort(histo.begin(), histo.end(), countComp);
        bool foundPlane = false;
        for (int r = 0; r < 180; r++) {
          if (histo[0].second < 4) {
            break;
          }
  
          uchar hue = static_cast<uchar>(histo[r].first);
          std::sort(histo.begin(), histo.end(), hueComp);
          std::vector<unsigned int> vertIndices = histoPoints[hue];
  
          int rd_num = rand() % vertIndices.size();
          unsigned int vert1Index = vertIndices[rd_num];
          vertIndices.erase(vertIndices.begin() + rd_num);
          
          rd_num = rand() % vertIndices.size();
          unsigned int vert2Index = vertIndices[rd_num];
          vertIndices.erase(vertIndices.begin() + rd_num);
  
          rd_num = rand() % vertIndices.size();
          unsigned int vert3Index = vertIndices[rd_num];
          vertIndices.erase(vertIndices.begin() + rd_num);
  
          glm::vec3 vert1Pos = glm::vec3(meshVertices[vert1Index], meshVertices[vert1Index + 1], meshVertices[vert1Index + 2]);
          glm::vec3 vert2Pos = glm::vec3(meshVertices[vert2Index], meshVertices[vert2Index + 1], meshVertices[vert2Index + 2]);
          glm::vec3 vert3Pos = glm::vec3(meshVertices[vert3Index], meshVertices[vert3Index + 1], meshVertices[vert3Index + 2]);
  
          glm::vec3 a = vert2Pos - vert1Pos;
          glm::vec3 b = vert3Pos - vert1Pos;
          glm::vec3 normal = glm::cross(a, b);
          float d = normal.x * -a.x + normal.y * -a.y + normal.z * -a.z;
          float denom = 1.0f / sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
  
  
          std::vector<unsigned int> planePointsIndices = {vert1Index, vert2Index, vert3Index};
          std::vector<unsigned int> remainingVertIndices;
          for (int i = 0; i < vertIndices.size(); i++) {
            unsigned int vertIndex = vertIndices[i];
            glm::vec3 pos = glm::vec3(meshVertices[vertIndex], meshVertices[vertIndex + 1], meshVertices[vertIndex + 2]);
            
            float D = abs(normal.x * pos.x + normal.y * pos.y + normal.z * pos.z + d) * denom;
            if (D < 0.05f) {
              planePointsIndices.push_back(vertIndex);
            }
            else {
              remainingVertIndices.push_back(vertIndex);
            }
          }
          
          // draw Quad
          if (planePointsIndices.size() < 1000) {
            continue;
          }
          else {
            histo[hue].second -= planePointsIndices.size();
            histoPoints[hue] = remainingVertIndices;
  
            for (int hue_x = hue-1; hue_x >= 0; hue_x--) {
              vertIndices = histoPoints[hue_x];
              int count = 0;
              for (int i = 0; i < vertIndices.size(); i++) {
                unsigned int vertIndex = vertIndices[i];
                glm::vec3 pos = glm::vec3(meshVertices[vertIndex], meshVertices[vertIndex + 1], meshVertices[vertIndex + 2]);
                
                float D = abs(normal.x * pos.x + normal.y * pos.y + normal.z * pos.z + d) * denom;
                if (D < 0.05f) {
                  planePointsIndices.push_back(vertIndex);
                  
                  histo[hue_x].second -= 1;
                  histoPoints[hue_x].erase(std::remove(histoPoints[hue_x].begin(), histoPoints[hue_x].end(), hue_x), histoPoints[hue_x].end());
                  count++;
                }
              }
              if (count == 0) {
                break;
              }
            }
  
            for (int hue_x = hue+1; hue_x <= 179; hue_x++) {
              vertIndices = histoPoints[hue_x];
              int count = 0;
              for (int i = 0; i < vertIndices.size(); i++) {
                unsigned int vertIndex = vertIndices[i];
                glm::vec3 pos = glm::vec3(meshVertices[vertIndex], meshVertices[vertIndex + 1], meshVertices[vertIndex + 2]);
                
                float D = abs(normal.x * pos.x + normal.y * pos.y + normal.z * pos.z + d) * denom;
                if (D < 0.05f) {
                  planePointsIndices.push_back(vertIndex);
                  
                  histo[hue_x].second -= 1;
                  histoPoints[hue_x].erase(std::remove(histoPoints[hue_x].begin(), histoPoints[hue_x].end(), hue_x), histoPoints[hue_x].end());
                  count++;
                }
              }
              if (count == 0) {
                break;
              }
            }
            
  
            std::cout << planePointsIndices.size() << "\n";          
            
            unsigned int vertIndex = planePointsIndices[0];
            glm::vec4 planeColor = glm::vec4(meshVertices[vertIndex + 3], meshVertices[vertIndex + 4], meshVertices[vertIndex + 5], meshVertices[vertIndex + 6]);
            for (int i = 0; i < planePointsIndices.size(); i++) {
              vertIndex = planePointsIndices[i];
  
              meshVertices2[vertIndex + 3] = planeColor.x;
              meshVertices2[vertIndex + 4] = planeColor.y;
              meshVertices2[vertIndex + 5] = planeColor.z;
              meshVertices2[vertIndex + 6] = planeColor.w;
            }
  
  
            float normalMag = glm::length(normal);
            glm::vec3 v = glm::cross(normal, glm::vec3(0.0f, 0.0f, 1.0f));
            float cosTheta = normal.z / normalMag;
            float sinTheta = glm::length(v) / normalMag;
  
            glm::vec3 k = glm::normalize(v);
            glm::mat3 K = glm::mat3(0.0f, -k.z, k.y,
                                    k.z, 0.0f, -k.x,
                                  -k.y, k.x, 0.0f);
            K = glm::transpose(K);
            glm::mat3 R = glm::mat3(1.0f) + sinTheta * K + (1.0f - cosTheta) * K * K;
  
            std::vector<std::pair<unsigned int, glm::vec3>> planePoints(planePointsIndices.size());
            for (int i = 0; i < planePointsIndices.size(); i++) {
              unsigned int vertIndex = planePointsIndices[i];
  
              glm::vec3 pos = glm::vec3(meshVertices[vertIndex*7], meshVertices[vertIndex*7 + 1], meshVertices[vertIndex*7 + 2]);
              glm::vec3 rotated = R * pos;
              
              planePoints[i].first = vertIndex;
              planePoints[i].second = rotated;
            }

            foundPlane = true;
            break;
          }
        }

        if (!foundPlane) {
          break;
        }
      }
      */


      glBindVertexArray(meshVAO);
      glBindBuffer(GL_ARRAY_BUFFER, meshVBO);
      glBufferData(GL_ARRAY_BUFFER, meshVertices.size() * sizeof(float), meshVertices.data(), GL_DYNAMIC_DRAW);

      glBindVertexArray(meshVAO2);
      glBindBuffer(GL_ARRAY_BUFFER, meshVBO2);
      glBufferData(GL_ARRAY_BUFFER, meshVertices2.size() * sizeof(float), meshVertices2.data(), GL_DYNAMIC_DRAW);

      glBindBuffer(GL_ARRAY_BUFFER, 0);
      glBindVertexArray(0);


      listener.release(frames);
    }


    meshShader.use();
    meshShader.setMat4("WVP", VP);
    //meshShader.setMat4("WVP", VP * glm::translate(glm::mat4(1.0f), glm::vec3(2.0f, 0.0f, 0.0f)));
    glBindVertexArray(meshVAO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, meshEBO);
    glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(meshIndices.size()), GL_UNSIGNED_INT, 0);

    //meshShader.setMat4("WVP", VP);
    //glBindVertexArray(meshVAO2);
    //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, meshEBO2);
    //glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(meshIndices.size()), GL_UNSIGNED_INT, 0);

    //meshShader.use();
    ////meshShader.setMat4("WVP", VP);
    //meshShader.setMat4("WVP", VP * glm::translate(glm::mat4(1.0f), glm::vec3(2.0f, 0.0f, 0.0f)));
    //for (int i = 0; i < numQuads; i++) {
    //  glBindVertexArray(quads[i].VAO);
    //  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, quads[i].EBO);
    //  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    //}
    

    cv::imshow("rgb", cvRGB);
    //cv::imshow("HDrgb", hdRGB);
    cv::imshow("hsv", cvHSV);
    std::vector<cv::Mat> channels;
    cv::split(cvHSV, channels);
    cv::imshow("hsv-hue", channels[0]);
    cv::imshow("depth", cvDepth/4500.0f);
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
  dev->stop();
  dev->close();
/// [stop]

  delete registration;

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
