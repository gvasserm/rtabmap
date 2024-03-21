
ADD_SUBDIRECTORY( BOWMapping )
#ADD_SUBDIRECTORY( BOWDemo )
ADD_SUBDIRECTORY( RBOWDemo )
ADD_SUBDIRECTORY( RgbdDataset )
ADD_SUBDIRECTORY( EurocDataset )
ADD_SUBDIRECTORY( NoEventsExample )
ADD_SUBDIRECTORY( NoEventsExample )

IF(TARGET rtabmap_gui)
   #ADD_SUBDIRECTORY( RGBDMapping )
   ADD_SUBDIRECTORY( WifiMapping )

   
ELSE()
   MESSAGE(STATUS "RTAB-Map GUI lib is not built, the RGBDMapping and WifiMapping examples will not be built...")
ENDIF()


