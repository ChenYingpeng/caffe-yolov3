
file(GLOB detectnetSources *.cpp)
file(GLOB detectnetIncludes *.h )

cuda_add_executable(detectnet ${detectnetSources})
target_link_libraries(detectnet sysDetectSpeed)
