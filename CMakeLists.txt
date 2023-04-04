cmake_minimum_required(VERSION 3.0.0)
project(yolo_ort)

set(ONNXRUNTIME_DIR /root/onnxruntime-linux-x64-1.11.0/)

find_package(OpenCV REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package (realsense2 REQUIRED)

include_directories( 
    ${OpenCV_INCLUDE_DIRS}/include
    ${realsense2_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/tracker/deepsort/include
    ${CMAKE_SOURCE_DIR}/tracker/bytetrack/include
    ${CMAKE_SOURCE_DIR}/detector/include
    ${CMAKE_SOURCE_DIR}/realsense/include

    )

add_executable( yolo_track
                main.cpp
                detector/src/detector.cpp
                detector/src/utils.cpp

                tracker/deepsort/src/FeatureTensor.cpp
                tracker/deepsort/src/model.cpp
                tracker/deepsort/src/kalmanfilter.cpp
                tracker/deepsort/src/linear_assignment.cpp
                tracker/deepsort/src/nn_matching.cpp
                tracker/deepsort/src/track.cpp
                tracker/deepsort/src/tracker.cpp
                tracker/deepsort/src/munkres.cpp
                tracker/deepsort/src/hungarianoper.cpp

                tracker/bytetrack/src/BytekalmanFilter.cpp
                tracker/bytetrack/src/BYTETracker.cpp
                tracker/bytetrack/src/lapjv.cpp
                tracker/bytetrack/src/STrack.cpp
                tracker/bytetrack/src/utils.cpp
                )
                
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

target_include_directories(yolo_track PRIVATE "${ONNXRUNTIME_DIR}/include")
# link_directories("${ONNXRUNTIME_DIR}/lib")
target_compile_features(yolo_track PRIVATE cxx_std_14)
target_link_libraries(yolo_track ${OpenCV_LIBS} Eigen3::Eigen ${realsense2_LIBRARY})

target_link_libraries(yolo_track "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
                
                
add_executable( test_track
                test_track.cpp
                detector/src/detector.cpp
                detector/src/utils.cpp

                tracker/deepsort/src/FeatureTensor.cpp
                tracker/deepsort/src/model.cpp
                tracker/deepsort/src/kalmanfilter.cpp
                tracker/deepsort/src/linear_assignment.cpp
                tracker/deepsort/src/nn_matching.cpp
                tracker/deepsort/src/track.cpp
                tracker/deepsort/src/tracker.cpp
                tracker/deepsort/src/munkres.cpp
                tracker/deepsort/src/hungarianoper.cpp

                tracker/bytetrack/src/BytekalmanFilter.cpp
                tracker/bytetrack/src/BYTETracker.cpp
                tracker/bytetrack/src/lapjv.cpp
                tracker/bytetrack/src/STrack.cpp
                tracker/bytetrack/src/utils.cpp
                )
target_include_directories(test_track PRIVATE "${ONNXRUNTIME_DIR}/include")
# link_directories("${ONNXRUNTIME_DIR}/lib")
target_compile_features(test_track PRIVATE cxx_std_14)
target_link_libraries(test_track ${OpenCV_LIBS} Eigen3::Eigen ${realsense2_LIBRARY})

target_link_libraries(test_track "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
