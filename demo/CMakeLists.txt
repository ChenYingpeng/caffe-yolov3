
file(GLOB demoSources *.cpp)
#file(GLOB detectnetIncludes *.h )

cuda_add_executable(demo ${demoSources})
target_link_libraries(demo yolov3-plugin)
