import os
import shutil
import time

def path(paths,env):
    proceed_path = []
    for path in paths:
        if env == 'windows':
            path_arr = path.split('\\')
            path = ''
            for path_part in path_arr:
                path += path_part + '/'
            proceed_path.append(path)
        else:
            path_arr = path.split('/')
            path = ''
            for path_part in path_arr:
                path += path_part + '/'
            proceed_path.append(path)
    return(proceed_path)

def clean(paths):
    for path in paths:
        if os.path.exists(path):
            shutil.rmtree(path)
            time.sleep(0.1)
            os.mkdir(path[:-1])
        else:
            os.mkdir(path[:-1])