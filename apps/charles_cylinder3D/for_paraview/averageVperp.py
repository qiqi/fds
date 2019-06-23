#### import the simple module from the paraview
from paraview.simple import *

pnglocation = 'D:\\figure_JFMcylinder\\temp6_change_w\\'

# create a new 'XML Unstructured Grid Reader'

z0_plane = XMLUnstructuredGridReader(FileName=['D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.270.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.271.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.272.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.273.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.274.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.275.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.276.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.277.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.278.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.279.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.280.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.281.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.282.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.283.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.284.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.285.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.286.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.287.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.288.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.289.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.290.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.291.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.292.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.293.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.294.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.295.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.296.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.297.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.298.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.299.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.300.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.301.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.302.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.303.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.304.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.305.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.306.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.307.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.308.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.309.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.310.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.311.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.312.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.313.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.314.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.315.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.316.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.317.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.318.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.319.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.320.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.321.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.322.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.323.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.324.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.325.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.326.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.327.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.328.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.329.vtu', 'D:\\figure_JFMcylinder\\temp6_change_w\\z0_plane_seg.330.vtu'])
z0_plane.CellArrayStatus = ['RHOU']

renderView1 = GetActiveViewOrCreate('RenderView')


# normalize by a new 'Calculator'
calculator1 = Calculator(Input=z0_plane)
calculator1.AttributeMode = 'Cell Data'
calculator1.ResultArrayName = 'rhou_mag'
calculator1.Function = 'sqrt(RHOU_X^2 + RHOU_Y^2 + RHOU_Z^2) / (1.1838 * 33.0 / 132000)'


# create a new 'Temporal Statistics'
temporalStatistics1 = TemporalStatistics(Input=calculator1)

# Properties modified on temporalStatistics1
temporalStatistics1.ComputeMinimum = 0
temporalStatistics1.ComputeMaximum = 0
temporalStatistics1.ComputeStandardDeviation = 0


PointData = CellDatatoPointData(Input=temporalStatistics1)
PointDataDisplay = Show(PointData, renderView1)
ColorBy(PointDataDisplay, ('POINTS', 'rhou_mag_average', 'Magnitude'))

# rescale color and/or opacity maps used to include current data range
PointDataDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
PointDataDisplay.SetScalarBarVisibility(renderView1, True)
renderView1.OrientationAxesVisibility = 0
newRHOULUT = GetColorTransferFunction('rhou_mag_average')
newRHOULUT.MapControlPointsToLogSpace()
# newRHOULUT.UseLogScale = 1
newRHOULUT.ApplyPreset('X Ray', True)
newRHOULUT.RescaleTransferFunction(0.3, 30)

rHOULUTColorBar = GetScalarBar(newRHOULUT, renderView1)
rHOULUTColorBar.WindowLocation = 'AnyLocation'
rHOULUTColorBar.TitleFontSize = 6
rHOULUTColorBar.LabelFontSize = 6
rHOULUTColorBar.Position = [0.03, 0.345]
rHOULUTColorBar.ScalarBarLength = 0.3
rHOULUTColorBar.LabelFormat = '%-#7.1f'
rHOULUTColorBar.RangeLabelFormat = '%-#7.1f'

# adjust lights
rHOULUTColorBar.TitleColor = [0.0, 0.0, 0.0]
rHOULUTColorBar.LabelColor = [0.0, 0.0, 0.0]
renderView1.LightSwitch = 0 
renderView1.UseLight = 0 

# current camera placement for renderView1
renderView1.CameraPosition = [0.0028, 0.0, 0.005]
renderView1.CameraFocalPoint = [0.0028, 0.0, 0.0]
renderView1.CameraParallelScale = 0.003
renderView1.CameraParallelProjection = 1

SaveScreenshot(pnglocation+'vperp_change_w_averaged.png', renderView1, ImageResolution=[1735, 1140])

