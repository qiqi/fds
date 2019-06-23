This the ref folder for the case where the parameter is the rotation speed, in round per time unit
the initial.les file is from file result.300000 in /U33_finerMesh folder, with the following settings:

# ==========================
# boundary conditions
# ==========================
left = CBC_UPT 33.0 0 0 101325. 298.15
right = NSCBC_OUTLET_P 101325.0 0.5 0.25e-3
cylinder = WALL_ADIABATIC
