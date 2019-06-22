#### import the simple module from the paraview
from paraview.simple import *

K_SEGMENTS = [300,]
# K_SEGMENTS = range(323,341)
pnglocation = 'D:\\figure_JFMcylinder\\temp6_change_w\\'

def plot(i_segment, pnglocation):
    print(i_segment)

    # create a new 'XML Unstructured Grid Reader'
    z0_plane = XMLUnstructuredGridReader(FileName=[pnglocation+'z0_plane_seg.'+str(i_segment)+'.vtu'])
    z0_plane.CellArrayStatus = ['RHO', 'RHOE', 'P', 'T', 'RHOU', 'U']

    # normalize by a new 'Calculator'
    calculator1 = Calculator(Input=z0_plane)
    calculator1.AttributeMode = 'Cell Data'
    calculator1.ResultArrayName = 'rhou'
    calculator1.Function = 'RHOU / (1.1838 * 33.0 / 132000)'

    renderView1 = GetActiveViewOrCreate('RenderView')
    # renderView1.ViewSize = [1735, 1140]

    # show data in view
    PointData = CellDatatoPointData(Input=calculator1)
    PointDataDisplay = Show(PointData, renderView1)
    ColorBy(PointDataDisplay, ('POINTS', 'rhou', 'Magnitude'))


    # rescale color and/or opacity maps used to include current data range
    PointDataDisplay.RescaleTransferFunctionToDataRange(True, False)

    # show color bar/color legend
    PointDataDisplay.SetScalarBarVisibility(renderView1, True)
    renderView1.OrientationAxesVisibility = 0
    newRHOULUT = GetColorTransferFunction('rhou')
    newRHOULUT.MapControlPointsToLogSpace()
    newRHOULUT.UseLogScale = 1
    newRHOULUT.ApplyPreset('X Ray', True)
    newRHOULUT.RescaleTransferFunction(0.3, 60)

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

    # camera placement 
    renderView1.CameraPosition = [0.0028, 0.0, 0.005]
    renderView1.CameraFocalPoint = [0.0028, 0.0, 0.0]
    renderView1.CameraParallelScale = 0.003
    renderView1.CameraParallelProjection = 1

    SaveScreenshot(pnglocation+'vperp_change_w_seg'+str(i_segment)+'.png', renderView1, ImageResolution=[1735, 1140])

for i_segment in K_SEGMENTS:
    plot(i_segment, pnglocation)
