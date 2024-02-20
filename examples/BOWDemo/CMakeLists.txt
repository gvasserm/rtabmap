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
	FIND_PACKAGE(RTABMap REQUIRED)
endif()

ADD_EXECUTABLE(bow_demo main.cpp)

if(internal)
  SET_TARGET_PROPERTIES( bow_mapping 
    PROPERTIES OUTPUT_NAME ${PROJECT_PREFIX}-bow_demo)
endif(internal)

