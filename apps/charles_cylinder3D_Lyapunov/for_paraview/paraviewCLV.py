#### import the simple module from the paraview
from paraview.simple import *

M_MODES = [0, 4, 16, 39]
K_SEGMENTS = [400,]
CLVlocation = 'D:\\figure_JFMcylinder\\temp1_CLV_finer\\'

def plot(i_segment, j_mode):
    print(i_segment, j_mode)

    # create a new 'XML Unstructured Grid Reader'
    pnglocation = CLVlocation + 'CLV' + str(j_mode)
    z0_plane = XMLUnstructuredGridReader(FileName=[pnglocation+'\\z0_plane_seg.'+str(i_segment)+'.vtu'])
    z0_plane.CellArrayStatus = ['RHO', 'RHOE', 'P', 'T', 'RHOU', 'U']

    # normalize by a new 'Calculator'
    calculator1 = Calculator(Input=z0_plane)
    calculator1.AttributeMode = 'Cell Data'
    calculator1.ResultArrayName = 'rhou'
    if j_mode == 0:
        calculator1.Function = 'RHOU/14.2'
    elif j_mode == 4:
        calculator1.Function = 'RHOU/0.04/0.5'
    elif j_mode == 16:
        calculator1.Function = 'RHOU/21/0.7'
    elif j_mode == 39:
        calculator1.Function = 'RHOU/16'
    else:
        pause;

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
    newRHOULUT.ApplyPreset('X Ray', True)
    newRHOULUT.RescaleTransferFunction(0.01, 1.00)

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

    SaveScreenshot(pnglocation+'\\..\\CLV_finer_'+str(j_mode)+'_'+str(i_segment)+'.png', renderView1, ImageResolution=[1735, 1140])

    Delete(z0_plane)
    del z0_plane

for i_segment in K_SEGMENTS:
    for j_mode in M_MODES:
        plot(i_segment, j_mode, CLVlocation)
