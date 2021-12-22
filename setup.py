import os

excludedDirs = ['.idea', "inspectionProfiles", "__pycache__", "todos"]


def folderStructure(dirname, path=os.path.sep):
    data = []
    for name in os.listdir(dirname):
        if name in excludedDirs:
            continue

        dct = {}
        dct['name'] = name
        dct['path'] = path + name

        full_path = os.path.join(dirname, name)

        if os.path.isfile(full_path):
            dct['type'] = 'file'
            continue
        elif os.path.isdir(full_path):
            dct['type'] = 'folder'
            dct['children'] = folderStructure(full_path, path=path + name + os.path.sep)

        data.append(dct)
    return data


projectStructure = folderStructure(".")
print(projectStructure)
import json

with open("projectStructure.json", "w") as outfile:
    json.dump(projectStructure, outfile, indent=4)
