
ADD_SUBDIRECTORY( BOWMapping )
#ADD_SUBDIRECTORY( BOWDemo )
ADD_SUBDIRECTORY( RBOWDemo )

IF(TARGET rtabmap_gui)
   #ADD_SUBDIRECTORY( RGBDMapping )
   ADD_SUBDIRECTORY( WifiMapping )
   ADD_SUBDIRECTORY( NoEventsExample )
ELSE()
   MESSAGE(STATUS "RTAB-Map GUI lib is not built, the RGBDMapping and WifiMapping examples will not be built...")
ENDIF()


