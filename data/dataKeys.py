class dreieckRasterHeaders:
    id = "id"
    x = "x"
    y = "y"
    z = "z"
    displacement_x = "displacement_x"
    displacement_y = "displacement_y"
    displacement_z = "displacement_z"
    epsilon_x = "epsilon_x"
    epsilon_y = "epsilon_y"
    epsilon_xy = "epsilon_xy"
    major_strain = "major_strain"
    minor_strain = "minor_strain"
    thickness_reduction = "thickness_reduction"

    allHeaders = [id, x, y, z,
                  displacement_x, displacement_y, displacement_z,
                  epsilon_x, epsilon_y, epsilon_xy,
                  major_strain, minor_strain,
                  thickness_reduction]


class viereckRasterHeaders:
    index_x = "index_x"
    index_y = "index_y"
    x_Koordinate = "x-Koordinate"
    y_Koordinate = "y-Koordinate"
    z_Koordinate = "z-Koordinate"
    verschiebungX = "Verschiebung x-Richtung [mm]"
    verschiebungY = "Verschiebung y-Richtung [mm]"
    verschiebungZ = "Verschiebung 7-Richtung [mm]"
    dehnungX = "Dehnung eps_x [%]"
    dehnungY = "Dehnung eps_y [%]"
    dehnungXY = "Dehnung eps_xy [log]"
    strainPhi1 = "strain_phi1 [log]"
    strainPhi2 = "strain_phi2 [log]"
    strainPhi3 = "strain_phi3 [log]"
    time = "time [sec]"

    allHeaders = [index_x, index_y,
                  x_Koordinate, y_Koordinate, z_Koordinate,
                  verschiebungX, verschiebungY, verschiebungZ,
                  dehnungX, dehnungY, dehnungXY,
                  strainPhi2, strainPhi3,
                  time]


class interpolatedDataHeaders:
    x = "x"
    y = "y"
    z = "z"
    xDisplacement = "xDisplacement"
    yDisplacement = "yDisplacement"
    zDisplacement = "zDisplacement"
    totalDisplacement = "totalDisplacement"

    allHeaders = [x, y, z, xDisplacement, yDisplacement, zDisplacement, totalDisplacement]
