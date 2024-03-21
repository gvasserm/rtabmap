
# ADD_EXECUTABLE(rgbd_dataset main.cpp)
  
# TARGET_LINK_LIBRARIES(rgbd_dataset rtabmap_core)

# SET_TARGET_PROPERTIES( rgbd_dataset 
#     PROPERTIES OUTPUT_NAME ${PROJECT_PREFIX}-rgbd_dataset)
    
# INSTALL(TARGETS rgbd_dataset
# 	RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT runtime
# 	BUNDLE DESTINATION "${CMAKE_BUNDLE_LOCATION}" COMPONENT runtime)


cmake_minimum_required(VERSION 3.5)

if(POLICY CMP0020)
	cmake_policy(SET CMP0020 NEW)
endif()

IF(DEFINED PROJECT_NAME)
	set(internal TRUE)
ENDIF(DEFINED PROJECT_NAME)

if(NOT internal)
	# external build
	PROJECT( MyProject )
	
	FIND_PACKAGE(RTABMap REQUIRED COMPONENTS gui)
endif()

IF(QT4_FOUND OR Qt5_FOUND OR Qt6_FOUND)
	SET(moc_srcs MapBuilder.h)
ENDIF()

ADD_EXECUTABLE(rgbd_dataset main.cpp ${moc_srcs})
	
TARGET_LINK_LIBRARIES(rgbd_dataset rtabmap::gui)

SET_TARGET_PROPERTIES(
	rgbd_dataset
	PROPERTIES
		AUTOUIC ON
		AUTOMOC ON
		AUTORCC ON
)

if(internal)
	SET_TARGET_PROPERTIES( rgbd_dataset 
		PROPERTIES OUTPUT_NAME ${PROJECT_PREFIX}-rgbd_dataset)
endif(internal)
