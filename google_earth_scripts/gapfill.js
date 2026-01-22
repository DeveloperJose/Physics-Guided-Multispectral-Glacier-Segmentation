var MIN_SCALE = 1/3;
var MAX_SCALE = 3;
var MIN_NEIGHBORS = 64;
/* Apply the USGS L7 Phase-2 Gap filling protocol, using a single kernel size. */
exports.GapFill = function(src) {
  var kernelSize = 5, upscale = true;
  var fill = ee.ImageCollection('LANDSAT/LE07/C01/T1_RT')
    //ee.ImageCollection('LANDSAT/LE07/C02/T2_L2')
    .filterDate("2000-01-01", "2000-12-31")
    .filter(ee.Filter.eq('WRS_ROW', src.get("WRS_ROW")))
    .filter(ee.Filter.eq('WRS_PATH', src.get("WRS_PATH")))
    .filter(ee.Filter.lt("CLOUD_COVER", 10))
    .sort('CLOUD_COVER').first();
  var kernel = ee.Kernel.square(kernelSize * 30, "meters", false)
  // Find the pixels common to both scenes.
  var common = src.mask().and(fill.mask())
  var fc = fill.updateMask(common)
  var sc = src.updateMask(common)
  // Find the primary scaling factors with a regression.
  // Interleave the bands for the regression.  This assumes the bands have the same names.
  var regress = fc.addBands(sc)
  regress = regress.select(regress.bandNames().sort())
  var ratio = 5
  if(upscale) {
    var fit = regress
      .reduceResolution(ee.Reducer.median(), false, 500)
      .reproject(regress.select(0).projection().scale(ratio, ratio))
      .reduceNeighborhood(ee.Reducer.linearFit().forEach(src.bandNames()), kernel, null, false)
      .unmask()
      .reproject(regress.select(0).projection().scale(ratio, ratio))
  } else {
    var fit = regress
      .reduceNeighborhood(ee.Reducer.linearFit().forEach(src.bandNames()), kernel, null, false)
  }
  var offset = fit.select(".*_offset")
  var scale = fit.select(".*_scale")
  // Find the secondary scaling factors using just means and stddev
  var reducer = ee.Reducer.mean().combine(ee.Reducer.stdDev(), null, true)
  if(upscale) {
    var src_stats = src
      .reduceResolution(ee.Reducer.median(), false, 500)
      .reproject(regress.select(0).projection().scale(ratio, ratio))
      .reduceNeighborhood(reducer, kernel, null, false)
      .reproject(regress.select(0).projection().scale(ratio, ratio))
    var fill_stats = fill
      .reduceResolution(ee.Reducer.median(), false, 500)
      .reproject(regress.select(0).projection().scale(ratio, ratio))
      .reduceNeighborhood(reducer, kernel, null, false)
      .reproject(regress.select(0).projection().scale(ratio, ratio))
  } else {
    var src_stats = src
      .reduceNeighborhood(reducer, kernel, null, false)
    var fill_stats = fill
      .reduceNeighborhood(reducer, kernel, null, false)
  }
  var scale2 = src_stats.select(".*stdDev").divide(fill_stats.select(".*stdDev"))
  var offset2 = src_stats.select(".*mean").subtract(fill_stats.select(".*mean").multiply(scale2))
  var invalid = scale.lt(MIN_SCALE).or(scale.gt(MAX_SCALE))
  scale = scale.where(invalid, scale2)
  offset = offset.where(invalid, offset2)
  // When all else fails, just use the difference of means as an offset.
  var invalid2 = scale.lt(MIN_SCALE).or(scale.gt(MAX_SCALE))
  scale = scale.where(invalid2, 1)
  offset = offset.where(invalid2, src_stats.select(".*mean").subtract(fill_stats.select(".*mean")))
  // Apply the scaling and mask off pixels that didn't have enough neighbors.
  var count = common.reduceNeighborhood(ee.Reducer.count(), kernel, null, true, "boxcar")
  var scaled = fill.multiply(scale).add(offset)
      .updateMask(count.gte(MIN_NEIGHBORS))
  return src.unmask(scaled, true).uint8()
}