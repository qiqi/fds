#### import the simple module from the paraview
from paraview.simple import *

pnglocation = 'D:\\figure_JFMcylinder\\temp1_CLV_finer\\'

# create a new 'XML Unstructured Grid Reader'

M_MODES = [0, 4, 16, 39]
K_SEGMENTS = range(350,451)

def plotAverage(m_mode):
    print('plotting mode '+ str(m_mode))

    files = ['D:\\figure_JFMcylinder\\temp1_CLV_finer\\CLV'+str(m_mode)+'\\z0_plane_seg.'+str(k)+'.vtu' for k in K_SEGMENTS]
    z0_plane = XMLUnstructuredGridReader(FileName=files)
    z0_plane.CellArrayStatus = ['RHOU']
    renderView1 = GetActiveViewOrCreate('RenderView')

    # normalize by a new 'Calculator'
    calculator1 = Calculator(Input=z0_plane)
    calculator1.AttributeMode = 'Cell Data'
    calculator1.ResultArrayName = 'rhou_mag'
    calculator1.Function = 'sqrt(RHOU_X^2 + RHOU_Y^2 + RHOU_Z^2)'
    if m_mode == 0:
        calculator1.Function = 'sqrt(RHOU_X^2 + RHOU_Y^2 + RHOU_Z^2)/9'
    elif m_mode == 4:
        calculator1.Function = 'sqrt(RHOU_X^2 + RHOU_Y^2 + RHOU_Z^2)/0.01/1.4' 
    elif m_mode == 16:
        calculator1.Function = 'sqrt(RHOU_X^2 + RHOU_Y^2 + RHOU_Z^2)/7/1.05'
    elif m_mode == 39:
        calculator1.Function = 'sqrt(RHOU_X^2 + RHOU_Y^2 + RHOU_Z^2)/9'
    else:
        pause;


    # only average data
    temporalStatistics1 = TemporalStatistics(Input=calculator1)
    temporalStatistics1.ComputeMinimum = 0
    temporalStatistics1.ComputeMaximum = 0
    temporalStatistics1.ComputeStandardDeviation = 0
    
    # point data
    PointData = CellDatatoPointData(Input=temporalStatistics1)
    PointDataDisplay = Show(PointData, renderView1)
    ColorBy(PointDataDisplay, ('POINTS', 'rhou_mag_average', 'Magnitude'))
    PointDataDisplay.RescaleTransferFunctionToDataRange(True, False)

    # show color bar/color legend
    PointDataDisplay.SetScalarBarVisibility(renderView1, True)
    renderView1.OrientationAxesVisibility = 0
    newRHOULUT = GetColorTransferFunction('rhou_mag_average')
    # newRHOULUT.MapControlPointsToLogSpace()
    # newRHOULUT.UseLogScale = 1
    newRHOULUT.ApplyPreset('X Ray', True)
    newRHOULUT.RescaleTransferFunction(0.01, 1.00)
   
    # color bar
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

    SaveScreenshot(pnglocation+'CLV_finer_averaged'+str(m_mode)+'.png', renderView1, ImageResolution=[1735, 1140])

for m in M_MODES:
    plotAverage(m)
