var glaciers_hkh = ee.FeatureCollection("users/bibekaryal7/Glacier_HKH");
var fishnet = ee.FeatureCollection("users/bibekaryal7/HKH/fishnet_clip");

// imports
var gapfill = require('users/bibekaryal7/get_hkh_tiff:gapfill.js');
var ids = require('users/bibekaryal7/get_hkh_tiff:ids.js');

// script parameters
var image_ids = ids.image_ids.l7_2005_ids,
    params = {max_cloud_cover: 10};

Map.addLayer(glaciers_hkh, {palette: '0000FF'}, 'Glaciers_HKH')

// loop over each image
for (var i = 0; i < image_ids.length; i++) {
  var image = ee.Image('LANDSAT/LE07/C01/T1_RT/'+image_ids[i]),
      date = ee.Date(image.get('system:time_start')),
      slc_failure_date = ee.Date(new Date(2003, 5, 31));
  if (date >= slc_failure_date){
      var gapfillImage = gapfill.GapFill(image);
  }
  // drop pansharpening band and bqa from l07
  var l07 = gapfillImage.clip(image.geometry()).select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6_VCID_1', 'B6_VCID_2', 'B7']).uint8();
  // save to drive folder
  if (i === 0)
    var all_images = ee.List([ee.Image(l07)])
  else{
    all_images = all_images.add(ee.Image(l07))
  }
}
all_images = ee.ImageCollection.fromImages(all_images);
//var combined_image = all_images.mosaic();
//Map.addLayer(combined_image, {bands: ['B5','B4','B2'], min:0, max:256}, 'all_image');

fishnet = fishnet.toList(fishnet.size())

for (var i = 0; i < fishnet.length().getInfo(); i++) {
  var feature = ee.Feature(fishnet.get(i)),
      geometry = feature.geometry();
  var image = all_images.filterBounds(geometry)
  var crs = image.first().select(['B1']).projection().crs();
  if (i == 0){
    var oldCrs = image.first().select(['B1']).projection().crs();
    var crsStr = oldCrs.getInfo();
  } else {
    var newCrs = image.first().select(['B1']).projection().crs();
    if(newCrs != oldCrs){
      var crsStr = newCrs.getInfo();
      oldCrs = newCrs;
    }
  }
  image = image.mosaic().clip(geometry);
  Export.image.toDrive({
    image: image,
    folder: 'Landsat7_2005',
    crs: crsStr,
    description: 'image'+i.toString(),
    maxPixels: 318080701,
    region: geometry,
    scale: 30
  });
}
