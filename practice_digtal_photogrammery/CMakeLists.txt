set(source_file ${source_file}
		absolute_orientation/absolute_orientation_kernal.cpp)

set(source_file ${source_file}
		absolute_orientation/absolute_orientation_kernal_1.cpp)

set(source_file ${source_file}
		dense_match/sgm.cpp)

set(source_file ${source_file}
		dense_match/sgbm.cpp)




add_library(pdp ${source_file})
target_link_libraries(pdp ${OpenCV_LIBS} ${dlib_LIBRARIES})