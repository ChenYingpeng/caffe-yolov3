
cmake_minimum_required(VERSION 2.8)
project(sysDetectSpeed)

# setup tensorRT flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")	# -std=gnu++11
set(BUILD_DEPS "YES" CACHE BOOL "If YES, will install dependencies into sandbox.  Automatically reset to NO after dependencies are installed.")

# if this is the first time running cmake, perform pre-build dependency install script (or if the user manually triggers re-building the dependencies)
#if( ${BUILD_DEPS} )
#	message("Launching pre-build dependency installer script...")
#
#	execute_process(COMMAND sh ../CMakePreBuild.sh
#				WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
#				RESULT_VARIABLE PREBUILD_SCRIPT_RESULT)

#	set(BUILD_DEPS "NO" CACHE BOOL "If YES, will install dependencies into sandbox.  #Automatically reset to NO after dependencies are installed." FORCE)
#	message("Finished installing dependencies")
#endif()


# Qt is used to load images (installed by ubuntu-desktop)
find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})
add_definitions(${QT_DEFINITIONS})


# setup CUDA
find_package(CUDA)

set(
	CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS}; 
    -O3 
	-gencode arch=compute_53,code=sm_53
	-gencode arch=compute_62,code=sm_62
)



# setup project output paths
set(PROJECT_OUTPUT_DIR  ${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_PROCESSOR})
set(PROJECT_INCLUDE_DIR ${PROJECT_OUTPUT_DIR}/include)

file(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR})
file(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)

message("-- system arch:  ${CMAKE_SYSTEM_PROCESSOR}")
message("-- output path:  ${PROJECT_OUTPUT_DIR}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)


# build C/C++ interface
include_directories(${PROJECT_INCLUDE_DIR} ${GIE_PATH}/include)
include_directories(/usr/include/gstreamer-1.0 /usr/lib/aarch64-linux-gnu/gstreamer-1.0/include /usr/include/glib-2.0 /usr/include/libxml2 /usr/lib/aarch64-linux-gnu/glib-2.0/include/ /usr/local/cuda-9.0/extras/CUPTI/include)
include_directories(${PROJECT_INCLUDE_DIR} 
	/home/nvidia/caffe/include 
	/home/nvidia/caffe/build/include 
	/usr/local/include  
	/usr/local/cuda/include  
	/usr/include
)


file(GLOB inferenceSources *.cpp *.cu )
file(GLOB inferenceIncludes *.h )

cuda_add_library(sysDetectSpeed SHARED ${inferenceSources})
target_link_libraries(sysDetectSpeed nvcaffe_parser nvinfer Qt4::QtGui GL gstreamer-1.0 gstapp-1.0)	#GLEW	# gstreamer-0.10 gstbase-0.10 gstapp-0.10 
target_link_libraries(sysDetectSpeed 
	/home/nvidia/caffe/build/lib/libcaffe.so  
	/usr/lib/aarch64-linux-gnu/libglog.so  
	/usr/lib/aarch64-linux-gnu/libgflags.so.2
    	/usr/lib/aarch64-linux-gnu/libboost_system.so  
	/usr/lib/aarch64-linux-gnu/libGLEW.so.1.13  
)


# transfer all headers to the include directory
foreach(include ${inferenceIncludes})
	message("-- Copying ${include}")
	configure_file(${include} ${PROJECT_INCLUDE_DIR} COPYONLY)
endforeach()


# create symbolic link for network data
execute_process( COMMAND "${CMAKE_COMMAND}" "-E" "create_symlink" "${PROJECT_SOURCE_DIR}/data/networks" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/networks" )
  
  
# copy image data
file(GLOB imageData ${PROJECT_SOURCE_DIR}/data/images/*)

foreach(image ${imageData})
	message("-- Copying ${image}")
	file(COPY ${image} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
	#configure_file(${include} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} COPYONLY)
endforeach()


add_subdirectory(detectnet)


#setup opencv
FIND_PACKAGE(OpenCV REQUIRED)

TARGET_LINK_LIBRARIES(sysDetectSpeed ${OpenCV_LIBS})

# install
foreach(include ${inferenceIncludes})
    install(FILES "${include}" DESTINATION include/sysDetectSpeed)
endforeach()

# install the shared library
install(TARGETS sysDetectSpeed DESTINATION lib/sysDetectSpeed EXPORT sysDetectSpeedConfig)

# install the cmake project, for importing
install(EXPORT sysDetectSpeedConfig DESTINATION share/sysDetectSpeed/cmake)

