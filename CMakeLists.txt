cmake_minimum_required(VERSION 2.8)
project(caffe-yolov3)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")	# -std=gnu++11

# setup CUDA
find_package(CUDA)

set(
	CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS}; 
    -O3 
	-gencode arch=compute_53,code=sm_53  #tegra tx1 
        -gencode arch=compute_61,code=sm_61  #gtx 1060
	-gencode arch=compute_62,code=sm_62  #tegra tx2
)

if(CUDA_VERSION_MAJOR GREATER 9)
	message("-- CUDA ${CUDA_VERSION_MAJOR} detected, enabling SM_72")
	set (
		CUDA_NVCC_FLAGS
		${CUDA_NVCC_FLAGS};
		-gencode arch=compute_72,code=sm_72 #xavier
                -gencode arch=compute_75,code=sm_75 #rtx2080 
	)
endif()

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
include_directories(${PROJECT_INCLUDE_DIR} 
	/home/chen/caffe/include 
	/home/chen/caffe/build/include 
)


file(GLOB inferenceSources src/*.cpp src/*.cu )
file(GLOB inferenceIncludes src/*.h )

cuda_add_library(yolov3-plugin SHARED ${inferenceSources})
target_link_libraries(yolov3-plugin 
	/home/chen/caffe/build/lib/libcaffe.so  
	/usr/lib/x86_64-linux-gnu/libglog.so  
	/usr/lib/x86_64-linux-gnu/libgflags.so.2
    	/usr/lib/x86_64-linux-gnu/libboost_system.so  
	/usr/lib/x86_64-linux-gnu/libGLEW.so.1.13  
)


# transfer all headers to the include directory
foreach(include ${inferenceIncludes})
	message("-- Copying ${include}")
	configure_file(${include} ${PROJECT_INCLUDE_DIR} COPYONLY)
endforeach()


# create symbolic link for network data
execute_process( COMMAND "${CMAKE_COMMAND}" "-E" "create_symlink" "${PROJECT_SOURCE_DIR}/data/networks" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/networks" )
  
add_subdirectory(demo)
add_subdirectory(eval)

#setup opencv
FIND_PACKAGE(OpenCV REQUIRED)

TARGET_LINK_LIBRARIES(yolov3-plugin ${OpenCV_LIBS})

# install
foreach(include ${inferenceIncludes})
    install(FILES "${include}" DESTINATION include/yolov3-plugin)
endforeach()

# install the shared library
install(TARGETS yolov3-plugin DESTINATION lib/yolov3-plugin EXPORT yolov3-pluginConfig)

# install the cmake project, for importing
install(EXPORT yolov3-pluginConfig DESTINATION share/yolov3-plugin/cmake)
