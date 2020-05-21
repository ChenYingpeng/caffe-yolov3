
file(GLOB evalSources *.cpp)
#file(GLOB detectnetIncludes *.h )

cuda_add_executable(eval ${evalSources})
target_link_libraries(eval yolov3-plugin)
