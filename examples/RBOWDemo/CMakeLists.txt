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

ADD_EXECUTABLE(rbow_demo main.cpp)
  
TARGET_LINK_LIBRARIES(rbow_demo rtabmap::gui)

SET_TARGET_PROPERTIES(
  rbow_demo
   PROPERTIES
	 AUTOUIC ON
    AUTOMOC ON
     AUTORCC ON
)

if(internal)
  SET_TARGET_PROPERTIES( rbow_demo 
    PROPERTIES OUTPUT_NAME ${PROJECT_PREFIX}-rbow_demo)
endif(internal)
