import os


def get_image_paths_from_dir(fdir):
    if not os.path.exists(fdir):
        print(f"The path {fdir} does not exits")
        return None
    flist = os.listdir(fdir)
    flist.sort()
    image_paths = []
    for i in range(0, len(flist)):
        fpath = os.path.join(fdir, flist[i])
        if os.path.isdir(fpath):
            image_paths.extend(get_image_paths_from_dir(fpath))
        else:
            image_paths.append(fpath)
    return image_paths
