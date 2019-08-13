from QgisStandalone import QgisStandalone

qgis = QgisStandalone(qgis_boundary_file='boundary_wgs84.shp',
                      qgis_install_path='C:\\Program Files (x86)\\QGIS 2.18',
                      qgis_input_shp_path='C:\\Users\\MaxWillx\\CMU course\\Paws Project\\PAWS_SoftWare\\Data\\all_shp_inconsistent\\wgs84',
                      qgis_output_shapefile_path='C:\\Users\\MaxWillx\\Desktop\\shapefiles123',
                      qgis_output_csv_path='C:\\Users\\MaxWillx\\Desktop\\csvfiles123',
                      qgis_input_layers=None)
qgis.run()

# default you don't neet to pass qgis_input_layers, you can also pass qgis_input_layers like this:
# qgis_input_layers = {
# 	'boundary_file': ['toy_boundary.shp'],
# 	'dist_layers': ['toy_patrol.shp', 'toy_poaching.shp', 'toy_road.shp', 'toy_river.shp'],
# 	'int_layers': ['toy_patrol.shp', 'toy_poaching.shp', 'toy_road.shp'],
# 	'raster_layers': ['toy_altitude.tif']
# }
#
