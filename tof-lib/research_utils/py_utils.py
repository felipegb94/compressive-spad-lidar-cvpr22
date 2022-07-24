'''
	Useful functions in python
'''
#### Standard Library Imports

#### Library imports

#### Local imports


def get_obj_functions(obj, filter_str=''):
    '''
        Get all callable functions of the object as a list of strings.
        filter_str only appends the functions that contain the filter_str 
    '''
    obj_funcs = []
    for func_name in dir(obj):
        if((callable(getattr(obj, func_name))) and (filter_str in func_name)):
            obj_funcs.append(func_name)
    return obj_funcs