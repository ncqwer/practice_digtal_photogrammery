set(source_file ${source_file}
		absolute_orientation/absolute_orientation_kernal.cpp)

set(source_file ${source_file}
		absolute_orientation/absolute_orientation_kernal_1.cpp)



add_library(pdp ${source_file})
target_link_libraries(pdp ${OpenCV_LIBS} ${dlib_LIBRARIES})