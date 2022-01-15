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

    allHeaders = [index_x, index_y, x_Koordinate, y_Koordinate, z_Koordinate]


class interpolatedDataHeaders:
    x = "x"
    y = "y"
    z = "z"
    xDisplacement = "xDisplacement"
    yDisplacement = "yDisplacement"
    zDisplacement = "zDisplacement"
    totalDisplacement = "totalDisplacement"

    allHeaders = [x, y, z, xDisplacement, yDisplacement, zDisplacement, totalDisplacement]
