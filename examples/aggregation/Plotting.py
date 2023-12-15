from optimism import VTKWriter

def output_mesh_and_fields(filename, mesh, scalarElemFields=[], scalarNodalFields=[], vectorNodalFields=[]):
    writer = VTKWriter.VTKWriter(mesh, baseFileName=filename)
    for nodeField in scalarNodalFields:
        writer.add_nodal_field(name=nodeField[0],
                               nodalData=nodeField[1],
                               fieldType=VTKWriter.VTKFieldType.SCALARS,
                               dataType=VTKWriter.VTKDataType.FLOAT)
    for nodeField in vectorNodalFields:
        writer.add_nodal_field(name=nodeField[0],
                               nodalData=nodeField[1],
                               fieldType=VTKWriter.VTKFieldType.VECTORS,
                               dataType=VTKWriter.VTKDataType.FLOAT)
    for elemField in scalarElemFields:
        writer.add_cell_field(name=elemField[0],
                              cellData=elemField[1],
                              fieldType=VTKWriter.VTKFieldType.SCALARS,
                              dataType=VTKWriter.VTKDataType.FLOAT)
    writer.write()
    print('write successful')